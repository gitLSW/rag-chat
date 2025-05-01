import argparse
import getpass
from typing import Optional
import sys

def main():
    parser = argparse.ArgumentParser(description="Create MongoDB read-only users for a company")
    parser.add_argument("company_id", help="The company ID to create users for")
    parser.add_argument("--password", help="Admin password (optional, will prompt if not provided)", required=False)
    args = parser.parse_args()

    # Get admin password if not provided
    admin_password: Optional[str] = args.password
    if admin_password is None:
        admin_password = getpass.getpass("Enter MongoDB admin password: ")

    # Set up environment (alternative to .env file)
    import os
    os.environ['MONGO_ADMIN_USER'] = 'admin'
    os.environ['MONGO_ADMIN_PASSWORD'] = admin_password

    try:
        # Create user manager
        user_manager = MongoDBUserManager()
        
        # Create users for all access levels (0-10)
        print(f"Creating read-only users for company: {args.company_id}")
        for access_level in range(1, 11):  # 1 through 10
            # Generate a secure password for each user
            password = generate_secure_password()
            success = user_manager.create_access_user(
                company_id=args.company_id,
                access_level=access_level,
                password=password
            )
            
            if success:
                print(f"Created user for level {access_level}:")
                print(f"  Username: llm_{args.company_id}_level{access_level}")
                print(f"  Password: {password}")
                print(f"  Access: docs_{access_level} to docs_10")
                print("-" * 40)
            else:
                print(f"Failed to create user for level {access_level}")
                sys.exit(1)
        
        print("All users created successfully!")
        print("\nImportant: Save these credentials securely!")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

def generate_secure_password(length: int = 16) -> str:
    """Generate a secure random password"""
    import secrets
    import string
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(secrets.choice(alphabet) for _ in range(length))

if __name__ == "__main__":
    main()