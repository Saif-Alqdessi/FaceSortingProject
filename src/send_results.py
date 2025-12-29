import os
import csv
import shutil
import requests
from pathlib import Path
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "Data", "output")
CSV_PATH = os.path.join(BASE_DIR, "attendees.csv")
REPORT_PATH = os.path.join(BASE_DIR, "execution_report.csv")
WEBHOOK_URL = "http://localhost:5678/webhook/f33ec700-f3d6-47be-b50e-fdd5ec2cc049"

def log_transaction(name, email, status, message):
    """
    Log a transaction to execution_report.csv.
    
    Args:
        name: Person's name
        email: Email address
        status: Status (SUCCESS/FAILED/SKIPPED)
        message: Additional message/error details
    """
    file_exists = os.path.exists(REPORT_PATH)
    
    try:
        with open(REPORT_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write headers if file is new
            if not file_exists:
                writer.writerow(['Timestamp', 'Name', 'Email', 'Status', 'Message'])
            
            # Write the transaction row
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow([timestamp, name, email, status, message])
            
    except Exception as e:
        # Don't crash the main process if logging fails
        print(f"⚠️  [WARNING] Failed to log transaction: {e}")

def load_attendees():
    """Load attendees.csv and return a dictionary mapping Name -> Email."""
    attendees = {}
    
    if not os.path.exists(CSV_PATH):
        print(f"❌ [ERROR] attendees.csv not found at: {CSV_PATH}")
        return attendees
    
    try:
        # Detect delimiter by reading first line
        with open(CSV_PATH, 'r', encoding='utf-8-sig') as f:
            first_line = f.readline()
            # Check which delimiter appears more (comma or semicolon)
            comma_count = first_line.count(',')
            semicolon_count = first_line.count(';')
            delimiter = ';' if semicolon_count > comma_count else ','
            print(f"🔍 [DEBUG] Detected delimiter: '{delimiter}' (commas: {comma_count}, semicolons: {semicolon_count})")
            f.seek(0)  # Reset to beginning
            
            reader = csv.DictReader(f, delimiter=delimiter)
            
            # Strip whitespace from fieldnames (column headers)
            reader.fieldnames = [field.strip() if field else field for field in reader.fieldnames]
            print(f"🔍 [DEBUG] Detected columns: {reader.fieldnames}")
            
            # Validate required columns exist
            if 'Name' not in reader.fieldnames or 'Email' not in reader.fieldnames:
                print(f"❌ [ERROR] Required columns 'Name' and 'Email' not found.")
                print(f"   Available columns: {reader.fieldnames}")
                return attendees
            
            row_count = 0
            for row in reader:
                row_count += 1
                # Strip whitespace from both keys and values
                cleaned_row = {k.strip(): v.strip() if v else '' for k, v in row.items()}
                
                name = cleaned_row.get('Name', '').strip()
                email = cleaned_row.get('Email', '').strip()
                
                # Skip rows without valid name or email
                if not name or not email:
                    print(f"⚠️  [SKIP] Row {row_count}: Missing Name or Email (Name='{name}', Email='{email}')")
                    continue
                
                # Basic email validation
                if '@' not in email:
                    print(f"⚠️  [SKIP] Row {row_count}: Invalid email format for '{name}' (Email='{email}')")
                    continue
                
                attendees[name] = email
            
            print(f"✅ [INFO] Loaded {len(attendees)} valid attendees from {row_count} rows in CSV")
            return attendees
            
    except UnicodeDecodeError as e:
        print(f"❌ [ERROR] Encoding issue with attendees.csv. Tried UTF-8-sig. Error: {e}")
        return attendees
    except Exception as e:
        print(f"❌ [ERROR] Failed to read attendees.csv: {e}")
        import traceback
        traceback.print_exc()
        return attendees

def zip_folder(folder_path, zip_path):
    """Compress a folder into a zip file."""
    try:
        shutil.make_archive(zip_path.replace('.zip', ''), 'zip', folder_path)
        zip_size = os.path.getsize(zip_path) / (1024 * 1024)  # Size in MB
        print(f"📦 [INFO] Created zip: {os.path.basename(zip_path)} ({zip_size:.2f} MB)")
        return True
    except Exception as e:
        print(f"❌ [ERROR] Failed to create zip for {folder_path}: {e}")
        return False

def send_to_webhook(email, zip_path):
    """Send zip file to n8n webhook via POST request.
    
    Returns:
        tuple: (success: bool, message: str)
    """
    if not os.path.exists(zip_path):
        error_msg = f"Zip file not found: {zip_path}"
        print(f"❌ [ERROR] {error_msg}")
        return False, error_msg
    
    try:
        with open(zip_path, 'rb') as f:
            files = {
                'attachment': (os.path.basename(zip_path), f, 'application/zip')
            }
            data = {
                'email': email,
                'subject': 'Your photos are ready',
                'message': f'Dear {email.split("@")[0]},\n\nYour event photos are ready! Please find them attached.\n\nBest regards,\nEvent Photo Team'
            }
            
            print(f"📤 [INFO] Sending to {email}...")
            response = requests.post(WEBHOOK_URL, files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                print(f"✅ [SUCCESS] Successfully sent photos to {email}")
                return True, "Photos sent successfully"
            else:
                error_msg = f"Webhook returned status {response.status_code}: {response.text[:200]}"
                print(f"⚠️  [WARNING] Webhook returned status {response.status_code} for {email}")
                print(f"   Response: {response.text[:200]}")
                return False, error_msg
                
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error: {str(e)}"
        print(f"❌ [ERROR] Network error while sending to {email}: {e}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Exception: {str(e)}"
        print(f"❌ [ERROR] Failed to send to {email}: {e}")
        return False, error_msg

def cleanup_zip(zip_path):
    """Delete zip file after sending."""
    try:
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print(f"🗑️  [INFO] Deleted zip file: {os.path.basename(zip_path)}")
            return True
    except Exception as e:
        print(f"⚠️  [WARNING] Failed to delete zip file {zip_path}: {e}")
        return False

def send_results():
    """Main function to process and send sorted photos."""
    print("🚀 [INFO] Starting photo distribution process...")
    print(f"📁 [INFO] Output directory: {OUTPUT_DIR}")
    print(f"🔗 [INFO] Webhook URL: {WEBHOOK_URL}\n")
    
    # Load attendees
    attendees = load_attendees()
    if not attendees:
        print("❌ [ERROR] No attendees loaded. Exiting.")
        return
    
    # Check output directory
    if not os.path.isdir(OUTPUT_DIR):
        print(f"❌ [ERROR] Output directory not found: {OUTPUT_DIR}")
        return
    
    # Get all folders in output directory
    folders = [f for f in os.listdir(OUTPUT_DIR) 
               if os.path.isdir(os.path.join(OUTPUT_DIR, f)) and f != 'Unknown']
    
    if not folders:
        print("⚠️  [WARNING] No folders found in output directory (excluding 'Unknown')")
        return
    
    print(f"📂 [INFO] Found {len(folders)} person folders to process\n")
    
    # Statistics
    stats = {
        'processed': 0,
        'sent': 0,
        'failed': 0,
        'not_found': 0
    }
    
    # Process each folder
    for folder_name in folders:
        folder_path = os.path.join(OUTPUT_DIR, folder_name)
        
        # Check if folder has any files
        files = [f for f in os.listdir(folder_path) 
                 if os.path.isfile(os.path.join(folder_path, f))]
        
        if not files:
            print(f"⚠️  [SKIP] {folder_name}: No files in folder")
            continue
        
        print(f"\n{'='*60}")
        print(f"👤 [PROCESSING] {folder_name} ({len(files)} photos)")
        print(f"{'='*60}")
        
        # Check if name exists in attendees
        if folder_name not in attendees:
            print(f"⚠️  [SKIP] {folder_name}: Not found in attendees.csv")
            log_transaction(folder_name, "N/A", "SKIPPED", "Name not found in attendees.csv")
            stats['not_found'] += 1
            continue
        
        email = attendees[folder_name]
        stats['processed'] += 1
        
        # Create zip file in temp location (same directory as folder)
        zip_filename = f"{folder_name}.zip"
        zip_path = os.path.join(OUTPUT_DIR, zip_filename)
        
        # Step 1: Create zip
        if not zip_folder(folder_path, zip_path):
            stats['failed'] += 1
            continue
        
        # Step 2: Send to webhook
        success, message = send_to_webhook(email, zip_path)
        if success:
            stats['sent'] += 1
            log_transaction(folder_name, email, "SUCCESS", "Photos sent successfully")
        else:
            stats['failed'] += 1
            log_transaction(folder_name, email, "FAILED", message)
        
        # Step 3: Cleanup zip file
        cleanup_zip(zip_path)
    
    # Final summary
    print(f"\n{'='*60}")
    print("📊 [SUMMARY] Distribution Complete")
    print(f"{'='*60}")
    print(f"✅ Processed: {stats['processed']}")
    print(f"📤 Successfully sent: {stats['sent']}")
    print(f"❌ Failed: {stats['failed']}")
    print(f"⚠️  Not in CSV: {stats['not_found']}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    send_results()

