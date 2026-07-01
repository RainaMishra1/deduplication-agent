import os
import json
import psycopg2
import logging
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field, validator
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.fuzz import partial_ratio
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="KYC Deduplication & Blacklist Engine",
    description="Automated Real-Time Risk & Identity Verification Pipeline",
    version="2.0.0"
)

# ============================================================================
# DATABASE CONNECTION POOL
# ============================================================================

db_pool = SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    host="localhost",
    database="postgres",
    user="postgres",
    password="Root",
    port="5432"
)

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = db_pool.getconn()
    try:
        yield conn
    finally:
        db_pool.putconn(conn)

# ============================================================================
# CONFIGURATION & WEIGHTS
# ============================================================================

class DedupeWeights:
    """Field weights for cumulative scoring"""
    PAN = 0.35
    AADHAAR_LAST4 = 0.25
    NAME = 0.15
    DOB = 0.12
    PHONE = 0.08
    ADDRESS = 0.05
    
    TOTAL = 1.0
    
    BLACKLIST_MULTIPLIER = 1.5
    FUZZY_NAME_THRESHOLD = 0.85
    ADDRESS_THRESHOLD = 0.70
    
    EXACT_MATCH = 1.0
    HIGH_CONFIDENCE = 0.85
    MEDIUM_CONFIDENCE = 0.70
    LOW_CONFIDENCE = 0.50
    WEAK_MATCH = 0.30

# ============================================================================
# PYDANTIC MODELS WITH VALIDATION
# ============================================================================

class ApplicantReads(BaseModel):
    """Applicant data model with validation"""
    name: str = Field(..., min_length=2, max_length=255, example="Rahul Sharma")
    dob: str = Field(..., example="1992-05-14")
    pan: str = Field(..., min_length=10, max_length=10, example="ABCDE1234F")
    phone: str = Field(..., min_length=10, max_length=15, example="9876543210")
    aadhaar_last4: str = Field(..., min_length=4, max_length=4, example="5678")
    address: str = Field(..., min_length=5, max_length=500, example="Flat 402, MG Road, Mumbai")
    
    @validator('dob')
    def validate_dob(cls, v):
        try:
            dob_date = datetime.strptime(v, '%Y-%m-%d')
            if dob_date > datetime.now():
                raise ValueError("Date of birth cannot be in future")
            return v
        except ValueError as e:
            raise ValueError(f"Invalid DOB format. Use YYYY-MM-DD: {str(e)}")
    
    @validator('pan')
    def validate_pan(cls, v):
        v = v.upper().strip()
        # PAN format: 5 letters, 4 digits, 1 letter
        pattern = r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$'
        if not re.match(pattern, v):
            raise ValueError(f"PAN must be 10 characters: 5 letters, 4 digits, 1 letter. Got: {v}")
        return v
    
    @validator('phone')
    def validate_phone(cls, v):
        cleaned = ''.join(filter(str.isdigit, v))
        if len(cleaned) < 10:
            raise ValueError("Phone number must have at least 10 digits")
        return cleaned[-10:]
    
    @validator('aadhaar_last4')
    def validate_aadhaar(cls, v):
        v = v.strip()
        if not v.isdigit():
            raise ValueError("Aadhaar last 4 must be digits")
        if len(v) != 4:
            raise ValueError("Aadhaar last 4 must be exactly 4 digits")
        return v
    
    def normalize(self) -> Dict[str, Any]:
        return {
            'name': self.name.strip().lower(),
            'dob': self.dob,
            'pan': self.pan.upper().strip(),
            'phone': ''.join(filter(str.isdigit, self.phone))[-10:],
            'aadhaar_last4': self.aadhaar_last4.strip(),
            'address': self.address.strip().lower()
        }

class KycEventPayload(BaseModel):
    wakes_on: str = Field("kyc.dedup_requested")
    reads: ApplicantReads

# ============================================================================
# FIELD COMPARISON FUNCTIONS
# ============================================================================

def compare_field(field_type: str, value1: str, value2: str) -> float:
    """Compare two field values and return similarity score (0.0 to 1.0)"""
    if not value1 or not value2:
        return 0.0
    
    v1 = str(value1).strip()
    v2 = str(value2).strip()
    
    if field_type == 'pan':
        v1_clean = re.sub(r'[^A-Z0-9]', '', v1.upper())
        v2_clean = re.sub(r'[^A-Z0-9]', '', v2.upper())
        return 1.0 if v1_clean == v2_clean else 0.0
    
    elif field_type == 'aadhaar_last4':
        v1_last4 = v1[-4:] if len(v1) >= 4 else v1
        v2_last4 = v2[-4:] if len(v2) >= 4 else v2
        return 1.0 if v1_last4 == v2_last4 else 0.0
    
    elif field_type == 'name':
        similarity = JaroWinkler.similarity(v1.lower(), v2.lower())
        return similarity if similarity >= DedupeWeights.FUZZY_NAME_THRESHOLD else 0.0
    
    elif field_type == 'dob':
        try:
            for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%d/%m/%Y']:
                try:
                    d1 = datetime.strptime(v1, fmt)
                    d2 = datetime.strptime(v2, fmt)
                    return 1.0 if d1 == d2 else 0.0
                except:
                    continue
            return 1.0 if v1 == v2 else 0.0
        except:
            return 1.0 if v1 == v2 else 0.0
    
    elif field_type == 'phone':
        v1_clean = re.sub(r'\D', '', v1)[-10:]
        v2_clean = re.sub(r'\D', '', v2)[-10:]
        return 1.0 if v1_clean == v2_clean else 0.0
    
    elif field_type == 'address':
        similarity = partial_ratio(v1.lower(), v2.lower()) / 100.0
        return similarity if similarity >= DedupeWeights.ADDRESS_THRESHOLD else 0.0
    
    return 0.0

# ============================================================================
# CUMULATIVE SCORING ENGINE
# ============================================================================

def calculate_cumulative_score(applicant: Dict, db_record: Dict, is_blacklist: bool = False) -> tuple:
    """Calculate cumulative confidence score for a single database record"""
    score = 0.0
    matched_fields = []
    
    field_weights = {
        'pan': DedupeWeights.PAN,
        'aadhaar_last4': DedupeWeights.AADHAAR_LAST4,
        'name': DedupeWeights.NAME,
        'dob': DedupeWeights.DOB,
        'phone': DedupeWeights.PHONE,
        'address': DedupeWeights.ADDRESS
    }
    
    for field, weight in field_weights.items():
        if field in applicant and field in db_record:
            match_score = compare_field(field, applicant[field], db_record[field])
            if match_score > 0:
                field_score = weight * match_score
                score += field_score
                matched_fields.append({
                    'field': field,
                    'similarity': round(match_score , 2),
                    'weight': round(weight, 2),
                    'contribution': round(field_score, 2)
                })
    
    if is_blacklist and score > 0:
        score = min(score * DedupeWeights.BLACKLIST_MULTIPLIER, 1.0)
    
    return score, matched_fields

# ============================================================================
# DATABASE QUERY FUNCTIONS
# ============================================================================

def search_blacklist(cursor, applicant: Dict) -> List[Dict]:
    """Search for applicant in blacklist_records table"""
    matches = []
    
    # Hard match: PAN or (Aadhaar+DOB)
    # NOTE: blacklist_records doesn't have 'address' column
    cursor.execute("""
        SELECT 
            'BLACKLIST_DB' as source,
            blacklist_id::text as id,
            name,
            reason,
            pan,
            aadhaar_last4,
            dob,
            phone
        FROM blacklist_records 
        WHERE pan = %s OR (aadhaar_last4 = %s AND dob = %s)
        LIMIT 10;
    """, (applicant['pan'], applicant['aadhaar_last4'], applicant['dob']))
    
    records = cursor.fetchall()
    for record in records:
        # Convert record to dict and add empty address for scoring
        record_dict = dict(record)
        record_dict['address'] = ''  # Blacklist doesn't have address
        score, matched_fields = calculate_cumulative_score(applicant, record_dict, is_blacklist=True)
        if score > 0:
            matches.append({
                'record': record_dict,
                'score': score,
                'matched_fields': matched_fields,
                'source': 'BLACKLIST'
            })
    
    # Soft match: Phone only (if no hard matches found)
    if not matches:
        cursor.execute("""
            SELECT 
                'BLACKLIST_DB' as source,
                blacklist_id::text as id,
                name,
                reason,
                pan,
                aadhaar_last4,
                dob,
                phone
            FROM blacklist_records 
            WHERE phone = %s
            LIMIT 5;
        """, (applicant['phone'],))
        
        records = cursor.fetchall()
        for record in records:
            record_dict = dict(record)
            record_dict['address'] = ''  # Blacklist doesn't have address
            score, matched_fields = calculate_cumulative_score(applicant, record_dict, is_blacklist=True)
            if score > 0 and score < 1.0:
                matches.append({
                    'record': record_dict,
                    'score': score,
                    'matched_fields': matched_fields,
                    'source': 'BLACKLIST'
                })
    
    return matches

def search_customers(cursor, applicant: Dict) -> List[Dict]:
    """Search for applicant in existing_customers table"""
    matches = []
    
    # Hard match: PAN or (Aadhaar+DOB)
    cursor.execute("""
        SELECT 
            'CUSTOMER_DB' as source,
            customer_id::text as id,
            name,
            address,
            pan,
            aadhaar_last4,
            dob,
            phone,
            status
        FROM existing_customers 
        WHERE pan = %s OR (aadhaar_last4 = %s AND dob = %s)
        LIMIT 10;
    """, (applicant['pan'], applicant['aadhaar_last4'], applicant['dob']))
    
    records = cursor.fetchall()
    for record in records:
        record_dict = dict(record)
        score, matched_fields = calculate_cumulative_score(applicant, record_dict, is_blacklist=False)
        if score > 0:
            matches.append({
                'record': record_dict,
                'score': score,
                'matched_fields': matched_fields,
                'source': 'CUSTOMER'
            })
    
    # Soft match: Phone only
    cursor.execute("""
        SELECT 
            'CUSTOMER_DB' as source,
            customer_id::text as id,
            name,
            address,
            pan,
            aadhaar_last4,
            dob,
            phone,
            status
        FROM existing_customers 
        WHERE phone = %s
        LIMIT 5;
    """, (applicant['phone'],))
    
    records = cursor.fetchall()
    for record in records:
        record_dict = dict(record)
        # Skip if already matched via hard criteria
        if any(m['record']['id'] == record_dict['id'] for m in matches):
            continue
        score, matched_fields = calculate_cumulative_score(applicant, record_dict, is_blacklist=False)
        if score > 0 and score < 0.90:
            matches.append({
                'record': record_dict,
                'score': score,
                'matched_fields': matched_fields,
                'source': 'CUSTOMER'
            })
    
    return matches

def fuzzy_name_search(cursor, applicant: Dict, existing_matches: List) -> List[Dict]:
    """Search for fuzzy name matches with same DOB"""
    matches = []
    
    cursor.execute("""
        SELECT 
            'CUSTOMER_DB' as source,
            customer_id::text as id,
            name,
            address,
            pan,
            aadhaar_last4,
            dob,
            phone,
            status
        FROM existing_customers 
        WHERE dob = %s
        LIMIT 20;
    """, (applicant['dob'],))
    
    candidates = cursor.fetchall()
    
    for candidate in candidates:
        candidate_dict = dict(candidate)
        # Skip if already matched
        if any(m['record']['id'] == candidate_dict['id'] for m in existing_matches):
            continue
        
        db_name = candidate_dict["name"].strip().lower()
        name_score = JaroWinkler.similarity(applicant['name'], db_name)
        
        if name_score >= DedupeWeights.FUZZY_NAME_THRESHOLD:
            score = name_score * DedupeWeights.NAME
            matches.append({
                'record': candidate_dict,
                'score': score,
                'matched_fields': [{
                    'field': 'name (fuzzy)',
                    'similarity': round(name_score * 100, 2),
                    'weight': round(DedupeWeights.NAME * 100, 2),
                    'contribution': round(score * 100, 2)
                }],
                'source': 'CUSTOMER'
            })
    
    return matches

# ============================================================================
# VERDICT DECISION ENGINE
# ============================================================================

def determine_verdict(confidence: float, is_blacklist: bool, has_matches: bool) -> dict:
    """Determine status and verdict based on confidence score"""
    
    if not has_matches:
        return {
            'status': 'CLEAR',
            'verdict': 'NO_MATCH',
            'action': 'Proceed with KYC'
        }
    
    if is_blacklist and confidence >= 0.70:
        return {
            'status': 'BLACKLISTED',
            'verdict': 'BLACKLISTED_FRAUD',
            'action': 'Immediate rejection required'
        }
    
    if confidence >= DedupeWeights.EXACT_MATCH:
        return {
            'status': 'REJECTED',
            'verdict': 'EXACT_MATCH',
            'action': 'Auto-reject application - Exact ID match found'
        }
    elif confidence >= DedupeWeights.HIGH_CONFIDENCE:
        return {
            'status': 'REJECTED',
            'verdict': 'HIGH_CONFIDENCE_MATCH',
            'action': 'Reject with manual verification'
        }
    elif confidence >= DedupeWeights.MEDIUM_CONFIDENCE:
        return {
            'status': 'REVIEW',
            'verdict': 'MEDIUM_CONFIDENCE_MATCH',
            'action': 'Send to manual review team'
        }
    elif confidence >= DedupeWeights.LOW_CONFIDENCE:
        return {
            'status': 'REVIEW',
            'verdict': 'LOW_CONFIDENCE_MATCH',
            'action': 'Request additional verification documents'
        }
    elif confidence >= DedupeWeights.WEAK_MATCH:
        return {
            'status': 'FLAGGED',
            'verdict': 'WEAK_MATCH',
            'action': 'Flag for monitoring, allow KYC'
        }
    else:
        return {
            'status': 'CLEAR',
            'verdict': 'NO_MATCH',
            'action': 'Proceed with KYC'
        }

# ============================================================================
# MAIN DEDUPLICATION API ENDPOINT
# ============================================================================

@app.post("/api/v1/kyc/dedup")
def process_dedup_api(event: KycEventPayload):
    """Process KYC deduplication request with cumulative confidence scoring"""
    
    # Validate and normalize applicant data
    applicant = event.reads.normalize()
    
    logger.info(f"Processing dedup request for: {applicant['name']} (PAN: {applicant['pan']})")
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Search for matches
            blacklist_matches = search_blacklist(cursor, applicant)
            customer_matches = search_customers(cursor, applicant)
            fuzzy_matches = fuzzy_name_search(cursor, applicant, customer_matches)
            
            # Combine all matches
            all_matches = blacklist_matches + customer_matches + fuzzy_matches
            
            # Sort by score descending
            all_matches.sort(key=lambda x: x['score'], reverse=True)
            
            # Determine if any blacklist matches exist
            has_blacklist = any(m['source'] == 'BLACKLIST' for m in all_matches)
            
            # Calculate final confidence (highest score)
            final_confidence = max([m['score'] for m in all_matches]) if all_matches else 0.0
            
            # Determine verdict
            verdict = determine_verdict(final_confidence, has_blacklist, bool(all_matches))
            
            # Log decision
            logger.info(
                f"Decision for {applicant['name']}: {verdict['verdict']} "
                f"(Confidence: {final_confidence:.2%})"
            )
            
            # Build response
            if all_matches:
                response = {
                    "emit": "dedup.match_found",
                    "output": {
                        "status": verdict['status'],
                        "verdict": verdict['verdict'],
                        "action": verdict['action'],
                        "confidence": round(final_confidence , 2),
                        "has_blacklist": has_blacklist,
                        "match_count": len(all_matches),
                        "match_summary": [
                            {
                                "id": m['record'].get('id', 'N/A'),
                                "name": m['record'].get('name', 'Unknown'),
                                "source": m['source'],
                                "score": round(m['score'], 2),
                                "matched_fields": [
                                    {
                                        "field": f['field'],
                                        "similarity": f['similarity'],
                                        "weight": f['weight'],
                                        "contribution": f.get('contribution', 0)
                                    }
                                    for f in m['matched_fields']
                                ]
                            }
                            for m in all_matches[:5]
                        ],
                        "matched_records": [m['record'] for m in all_matches[:5]]
                    }
                }
            else:
                response = {
                    "emit": "dedup.clear",
                    "output": {
                        "status": "CLEAR",
                        "verdict": "NO_MATCH",
                        "action": "Proceed with KYC",
                        "confidence": 0.0,
                        "has_blacklist": False,
                        "match_count": 0,
                        "match_summary": [],
                        "matched_records": []
                    }
                }
            
            return response
            
    except psycopg2.Error as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================

@app.get("/api/v1/health")
def health_check():
    """Health check endpoint"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service Unavailable: {str(e)}")

# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "service": "KYC Deduplication & Blacklist Engine",
        "version": "2.0.0",
        "endpoints": {
            "deduplication": "/api/v1/kyc/dedup (POST)",
            "health": "/api/v1/health (GET)",
            "docs": "/docs (GET)",
            "redoc": "/redoc (GET)"
        }
    }

# ============================================================================
# RUN THE APPLICATION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )