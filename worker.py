import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.fuzz import partial_ratio
import re
from datetime import datetime

# 1. Database Connection Helper
def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        database="postgres",      
        user="postgres",          
        password="Root",          
        port="5432"
    )

# 2. Field Weight Configuration
class DedupeWeights:
    # Primary identifiers (highest uniqueness)
    PAN = 0.35
    AADHAAR_LAST4 = 0.25
    
    # Secondary identifiers (medium uniqueness)
    NAME = 0.15
    DOB = 0.12
    
    # Tertiary identifiers (low uniqueness, often change)
    PHONE = 0.08
    ADDRESS = 0.05
    
    # Total = 1.0 (100%)
    
    # Thresholds
    BLACKLIST_MULTIPLIER = 1.5  # 50% penalty for blacklist matches
    FUZZY_NAME_THRESHOLD = 0.85  # Minimum similarity for name match
    ADDRESS_THRESHOLD = 0.70     # Minimum similarity for address match
    CONFIDENCE_HARD_MATCH = 1.0   # Exact PAN/Aadhaar match
    CONFIDENCE_SOFT_MATCH = 0.90  # Phone-only match

# 3. Field Comparison Functions
def compare_fields(field_type, value1, value2):
    """
    Returns match confidence (0.0 to 1.0) for a specific field.
    """
    if not value1 or not value2:
        return 0.0
    
    # Normalize values
    v1 = str(value1).strip()
    v2 = str(value2).strip()
    
    if field_type == 'pan':
        # PAN: 5 letters, 4 digits, 1 letter (case insensitive)
        v1_clean = re.sub(r'[^A-Z0-9]', '', v1.upper())
        v2_clean = re.sub(r'[^A-Z0-9]', '', v2.upper())
        return 1.0 if v1_clean == v2_clean else 0.0
    
    elif field_type == 'aadhar_last4':
        # Compare only last 4 digits
        v1_last4 = v1[-4:] if len(v1) >= 4 else v1
        v2_last4 = v2[-4:] if len(v2) >= 4 else v2
        return 1.0 if v1_last4 == v2_last4 else 0.0
    
    elif field_type == 'name':
        # Fuzzy match with Jaro-Winkler
        similarity = JaroWinkler.similarity(v1.lower(), v2.lower())
        return similarity if similarity >= DedupeWeights.FUZZY_NAME_THRESHOLD else 0.0
    
    elif field_type == 'dob':
        # Parse and compare exact dates
        try:
            # Try multiple date formats
            for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%d/%m/%Y']:
                try:
                    d1 = datetime.strptime(v1, fmt)
                    d2 = datetime.strptime(v2, fmt)
                    return 1.0 if d1 == d2 else 0.0
                except:
                    continue
            # Fallback to string comparison
            return 1.0 if v1 == v2 else 0.0
        except:
            return 1.0 if v1 == v2 else 0.0
    
    elif field_type == 'phone':
        # Remove non-numeric and compare last 10 digits
        v1_clean = re.sub(r'\D', '', v1)[-10:]
        v2_clean = re.sub(r'\D', '', v2)[-10:]
        return 1.0 if v1_clean == v2_clean else 0.0
    
    elif field_type == 'address':
        # Fuzzy match with partial ratio
        similarity = partial_ratio(v1.lower(), v2.lower()) / 100.0
        return similarity if similarity >= DedupeWeights.ADDRESS_THRESHOLD else 0.0
    
    return 0.0

# 4. Calculate Cumulative Score for a Single Record
def calculate_record_score(applicant, db_record, is_blacklist=False):
    """
    Calculate cumulative confidence score for a single database record.
    Returns: (score, matched_fields)
    """
    score = 0.0
    matched_fields = []
    
    # Calculate score for each field
    field_weights = {
        'pan': DedupeWeights.PAN,
        'aadhar_last4': DedupeWeights.AADHAAR_LAST4,
        'name': DedupeWeights.NAME,
        'dob': DedupeWeights.DOB,
        'phone': DedupeWeights.PHONE,
        'address': DedupeWeights.ADDRESS
    }
    
    for field, weight in field_weights.items():
        if field in applicant and field in db_record:
            match_score = compare_fields(field, applicant[field], db_record[field])
            if match_score > 0:
                score += weight * match_score
                matched_fields.append({
                    'field': field,
                    'similarity': round(match_score * 100, 2),
                    'weight': round(weight * 100, 2)
                })
    
    # Apply blacklist multiplier penalty
    if is_blacklist and score > 0:
        score = min(score * DedupeWeights.BLACKLIST_MULTIPLIER, 1.0)
    
    return score, matched_fields

# 5. Main Deduplication Engine
def process_dedup(event_payload):
    applicant = event_payload["reads"]
    
    # Normalize inputs
    input_name = applicant["name"].strip().lower()
    input_dob = applicant["dob"]
    input_pan = applicant["pan"].strip().upper()
    input_phone = ''.join(filter(str.isdigit, applicant["phone"]))[-10:] 
    input_aadhaar = applicant["aadhaar_last4"].strip()

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:
        all_matches = []
        final_confidence = 0.0
        source = None
        match_reason = []
        
        # ------------------------------------------------------------------
        # STAGE 1: CHECK BLACKLIST DATABASE
        # ------------------------------------------------------------------
        
        # Check 1: Blacklist with PAN or Aadhaar+DOB
        cursor.execute("""
            SELECT 
                'BLACKLIST_DB' as source, 
                blacklist_id as id, 
                name, 
                reason,
                pan,
                aadhaar_last4,
                dob,
                phone,
                address
            FROM blacklist_records 
            WHERE pan = %s OR (aadhaar_last4 = %s AND dob = %s);
        """, (input_pan, input_aadhaar, input_dob))
        
        bl_hard_matches = cursor.fetchall()
        
        for record in bl_hard_matches:
            score, matched_fields = calculate_record_score(applicant, dict(record), is_blacklist=True)
            if score > 0:
                all_matches.append({
                    'record': dict(record),
                    'score': score,
                    'matched_fields': matched_fields,
                    'source': 'BLACKLIST'
                })
                final_confidence = max(final_confidence, score)
                match_reason.append(f"Blacklist match: {record['reason']}")
                source = 'BLACKLIST'
        
        # Check 2: Blacklist with Phone Only
        cursor.execute("""
            SELECT 
                'BLACKLIST_DB' as source, 
                blacklist_id as id, 
                name, 
                reason,
                pan,
                aadhaar_last4,
                dob,
                phone,
                address
            FROM blacklist_records 
            WHERE phone = %s;
        """, (input_phone,))
        
        bl_soft_matches = cursor.fetchall()
        
        for record in bl_soft_matches:
            score, matched_fields = calculate_record_score(applicant, dict(record), is_blacklist=True)
            if score > 0 and score < 1.0:  # Only if not already a hard match
                all_matches.append({
                    'record': dict(record),
                    'score': score,
                    'matched_fields': matched_fields,
                    'source': 'BLACKLIST'
                })
                final_confidence = max(final_confidence, score)
                match_reason.append(f"Blacklist phone match: {record['reason']}")
                source = 'BLACKLIST'
        
        # ------------------------------------------------------------------
        # STAGE 2: CHECK EXISTING CUSTOMER DATABASE
        # ------------------------------------------------------------------
        
        # Check 3: Customer with PAN or Aadhaar+DOB
        cursor.execute("""
            SELECT 
                'CUSTOMER_DB' as source, 
                customer_id as id, 
                name, 
                address,
                pan,
                aadhaar_last4,
                dob,
                phone
            FROM existing_customers 
            WHERE pan = %s OR (aadhaar_last4 = %s AND dob = %s);
        """, (input_pan, input_aadhaar, input_dob))
        
        cust_hard_matches = cursor.fetchall()
        
        for record in cust_hard_matches:
            score, matched_fields = calculate_record_score(applicant, dict(record), is_blacklist=False)
            if score > 0:
                all_matches.append({
                    'record': dict(record),
                    'score': score,
                    'matched_fields': matched_fields,
                    'source': 'CUSTOMER'
                })
                final_confidence = max(final_confidence, score)
                match_reason.append("Customer match via PAN or Aadhaar+DOB")
                source = 'CUSTOMER'
        
        # Check 4: Customer with Phone Only
        cursor.execute("""
            SELECT 
                'CUSTOMER_DB' as source, 
                customer_id as id, 
                name, 
                address,
                pan,
                aadhaar_last4,
                dob,
                phone
            FROM existing_customers 
            WHERE phone = %s;
        """, (input_phone,))
        
        cust_soft_matches = cursor.fetchall()
        
        for record in cust_soft_matches:
            score, matched_fields = calculate_record_score(applicant, dict(record), is_blacklist=False)
            if score > 0 and score < 0.90:  # Avoid duplicate high scores
                all_matches.append({
                    'record': dict(record),
                    'score': score,
                    'matched_fields': matched_fields,
                    'source': 'CUSTOMER'
                })
                final_confidence = max(final_confidence, score)
                match_reason.append("Customer match via phone")
                source = 'CUSTOMER'
        
        # ------------------------------------------------------------------
        # STAGE 3: FUZZY NAME MATCHING (Same DOB)
        # ------------------------------------------------------------------
        cursor.execute("""
            SELECT 
                'CUSTOMER_DB' as source, 
                customer_id as id, 
                name, 
                address,
                pan,
                aadhaar_last4,
                dob,
                phone
            FROM existing_customers 
            WHERE dob = %s;
        """, (input_dob,))
        
        dob_candidates = cursor.fetchall()
        
        for candidate in dob_candidates:
            # Skip if this candidate already matched via hard/soft criteria
            if any(m['record']['id'] == candidate['id'] for m in all_matches):
                continue
                
            db_name = candidate["name"].strip().lower()
            name_score = JaroWinkler.similarity(input_name, db_name)
            
            if name_score >= DedupeWeights.FUZZY_NAME_THRESHOLD:
                # Calculate partial score for fuzzy name match
                score = name_score * DedupeWeights.NAME
                
                all_matches.append({
                    'record': dict(candidate),
                    'score': score,
                    'matched_fields': [{
                        'field': 'name (fuzzy)',
                        'similarity': round(name_score * 100, 2),
                        'weight': round(DedupeWeights.NAME * 100, 2)
                    }],
                    'source': 'CUSTOMER'
                })
                final_confidence = max(final_confidence, score)
                match_reason.append(f"Fuzzy name match: {round(name_score * 100, 2)}% similarity")
                source = 'CUSTOMER'

        # ------------------------------------------------------------------
        # STAGE 4: DETERMINE FINAL VERDICT
        # ------------------------------------------------------------------
        
        # Sort matches by score descending
        all_matches.sort(key=lambda x: x['score'], reverse=True)
        
        # Determine status and confidence
        if final_confidence >= 1.0:
            status = "REJECTED"
            verdict = "EXACT_MATCH"
            confidence = 1.0
        elif final_confidence >= 0.85:
            status = "REJECTED"
            verdict = "HIGH_CONFIDENCE_MATCH"
            confidence = final_confidence
        elif final_confidence >= 0.70:
            status = "REVIEW"
            verdict = "MEDIUM_CONFIDENCE_MATCH"
            confidence = final_confidence
        elif final_confidence >= 0.50:
            status = "REVIEW"
            verdict = "LOW_CONFIDENCE_MATCH"
            confidence = final_confidence
        elif final_confidence >= 0.30:
            status = "FLAGGED"
            verdict = "WEAK_MATCH"
            confidence = final_confidence
        else:
            status = "CLEAR"
            verdict = "NO_MATCH"
            confidence = 0.0
        
        # Special case: Blacklist match with ANY confidence
        if source == 'BLACKLIST' and final_confidence > 0:
            status = "REJECTED"
            verdict = "BLACKLISTED"
            confidence = max(confidence, 0.70)  # Minimum 70% for blacklist
        
        # Build response
        response = {
            "emit": "dedup.match_found" if all_matches else "dedup.clear",
            "output": {
                "status": status,
                "verdict": verdict,
                "confidence": round(confidence * 100, 2),  # Convert to percentage
                "source": source,
                "match_reason": match_reason if match_reason else ["No matches found"],
                "matched_records": [m['record'] for m in all_matches[:5]],  # Top 5
                "detailed_matches": all_matches[:5] if all_matches else []  # Full details
            }
        }
        
        return response

    except Exception as e:
        print(f"Error in deduplication: {str(e)}")
        return {
            "emit": "dedup.error",
            "output": {
                "status": "ERROR",
                "verdict": "SYSTEM_ERROR",
                "confidence": 0.0,
                "error": str(e)
            }
        }
    finally:
        cursor.close()
        conn.close()

# 6. Test Scenarios
if __name__ == "__main__":
    # Test data
    test_scenarios = {
        "1. Hard Blacklist Match": {
            "reads": {
                "name": "Nirav Modi Gupta", 
                "dob": "1975-02-23", 
                "pan": "BLIST6666F", 
                "phone": "9006006677", 
                "aadhaar_last4": "9006", 
                "address": "Hub"
            }
        },
        "2. Soft Phone Blacklist": {
            "reads": {
                "name": "John Doe", 
                "dob": "2000-01-01", 
                "pan": "NEWPAN1111", 
                "phone": "9006006677", 
                "aadhaar_last4": "0000", 
                "address": "Unknown"
            }
        },
        "3. Hard Customer Match": {
            "reads": {
                "name": "Priyanka Das", 
                "dob": "1993-05-18", 
                "pan": "DEFGHI4444D", 
                "phone": "9222333444", 
                "aadhaar_last4": "4004", 
                "address": "Kolkata"
            }
        },
        "4. Fuzzy Name Match": {
            "reads": {
                "name": "Sidharth Malotra", 
                "dob": "1990-12-25", 
                "pan": "NEWPAN9999", 
                "phone": "9000000000", 
                "aadhaar_last4": "0000", 
                "address": "Kochi"
            }
        },
        "5. Clean User": {
            "reads": {
                "name": "Sachin Tendulkar", 
                "dob": "1973-04-24", 
                "pan": "SRTPA1111A", 
                "phone": "9999988888", 
                "aadhaar_last4": "1973", 
                "address": "Mumbai"
            }
        },
        "6. Partial Match Multiple Fields": {
            "reads": {
                "name": "Priya Dasgupta", 
                "dob": "1993-05-18", 
                "pan": "DEFGHI4444D", 
                "phone": "9222333444", 
                "aadhaar_last4": "5005", 
                "address": "Kolkata"
            }
        }
    }

    print("="*60)
    print("DEDUPLICATION ENGINE WITH CUMULATIVE SCORING")
    print("="*60)
    
    for name, payload in test_scenarios.items():
        print(f"\n▶ {name}")
        print("-" * 40)
        result = process_dedup(payload)
        print(f"Emit: {result['emit']}")
        print(f"Status: {result['output']['status']}")
        print(f"Verdict: {result['output']['verdict']}")
        print(f"Confidence: {result['output']['confidence']}%")
        print(f"Reasons: {', '.join(result['output']['match_reason'])}")
        
        if result['output']['matched_records']:
            print(f"Matched Records: {len(result['output']['matched_records'])} found")
            for i, record in enumerate(result['output']['matched_records'][:2], 1):
                print(f"  {i}. ID: {record.get('id', 'N/A')} - {record.get('name', 'Unknown')}")
        
        if result['output']['detailed_matches']:
            print("Detailed Scores:")
            for match in result['output']['detailed_matches'][:2]:
                print(f"  Score: {round(match['score'] * 100, 2)}%")
                for field in match.get('matched_fields', []):
                    print(f"    - {field['field']}: {field['similarity']}% (weight: {field['weight']}%)")
        
        print("-" * 40)