#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notion Workspace Reorganization Script
Fixes orphaned pages and creates new hub structure
ZERO DELETIONS - Only moves and quarantines
"""

import requests
import json
from datetime import datetime
import sys
import io
import os

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Notion API Configuration
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
if not NOTION_TOKEN:
    print("Error: NOTION_TOKEN environment variable is not set.")
    sys.exit(1)
NOTION_VERSION = "2022-06-28"
BASE_URL = "https://api.notion.com/v1"

HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": NOTION_VERSION
}

# Known Page IDs
CODE_REVIEW_AGENT = "281e8586-e4a2-813e-ac77-e0c43f4344fb"
INFRASTRUCTURE_HUB = "281e8586-e4a2-81b0-bbc6-f5a413c58322"
MATH_LEARNING_HUB = "281e8586-e4a2-81b2-b83c-cf6e964849fc"
COMMAND_CENTER = "281e8586-e4a2-81af-997a-c921703cc2d8"

# Orphaned pages that need fixing
ORPHANED_CODE_REVIEW_PAGES = [
    "8252fdbf-c77f-4848-bd97-0efc65811c11",  # Implementation Plan
    "d241d7e2-0e18-40d8-8111-f1ee3f076d74",  # Workflow Architecture
    "ef0069c4-e6cc-4558-b074-339fb0035dd8"   # Session Log
]

DUPLICATE_TO_QUARANTINE = "281e8586-e4a2-810b-bd93-c5ccf6f45ad1"  # Duplicate Implementation Plan

# Change tracking
changes_log = []

def log_change(action, page_id, page_title, details):
    """Log all changes for the report"""
    timestamp = datetime.now().isoformat()
    changes_log.append({
        "timestamp": timestamp,
        "action": action,
        "page_id": page_id,
        "page_title": page_title,
        "details": details
    })
    print(f"[{action}] {page_title} - {details}")

def create_page(title, emoji, parent_id=None):
    """Create a new page at workspace root or under a parent"""
    payload = {
        "parent": {"type": "workspace", "workspace": True} if parent_id is None else {"type": "page_id", "page_id": parent_id},
        "properties": {
            "title": [
                {
                    "type": "text",
                    "text": {"content": title}
                }
            ]
        }
    }

    if emoji:
        payload["icon"] = {"type": "emoji", "emoji": emoji}

    response = requests.post(f"{BASE_URL}/pages", headers=HEADERS, json=payload)

    if response.status_code == 200:
        page = response.json()
        log_change("CREATE", page["id"], title, f"Created new hub/page")
        return page["id"]
    else:
        print(f"ERROR creating page '{title}': {response.status_code} - {response.text}")
        return None

def move_page(page_id, new_parent_id):
    """Move a page to a new parent"""
    # First get current page info
    page_response = requests.get(f"{BASE_URL}/pages/{page_id}", headers=HEADERS)
    if page_response.status_code != 200:
        print(f"ERROR fetching page {page_id}: {page_response.text}")
        return False

    page = page_response.json()
    page_title = page.get("properties", {}).get("title", {}).get("title", [{}])[0].get("plain_text", "Unknown Title")

    # Move the page
    payload = {
        "parent": {"type": "page_id", "page_id": new_parent_id}
    }

    response = requests.patch(f"{BASE_URL}/pages/{page_id}", headers=HEADERS, json=payload)

    if response.status_code == 200:
        log_change("MOVE", page_id, page_title, f"Moved to parent {new_parent_id}")
        return True
    else:
        print(f"ERROR moving page {page_id}: {response.status_code} - {response.text}")
        return False

def quarantine_page(page_id, quarantine_parent_id):
    """Move page to quarantine and prefix title with [DUPLICATE?]"""
    # Get current page
    page_response = requests.get(f"{BASE_URL}/pages/{page_id}", headers=HEADERS)
    if page_response.status_code != 200:
        print(f"ERROR fetching page {page_id}: {page_response.text}")
        return False

    page = page_response.json()
    current_title_array = page.get("properties", {}).get("title", {}).get("title", [])
    current_title = current_title_array[0].get("plain_text", "Unknown") if current_title_array else "Unknown"

    # Update title and move to quarantine
    new_title = f"[DUPLICATE?] {current_title}"
    payload = {
        "parent": {"type": "page_id", "page_id": quarantine_parent_id},
        "properties": {
            "title": [
                {
                    "type": "text",
                    "text": {"content": new_title}
                }
            ]
        }
    }

    response = requests.patch(f"{BASE_URL}/pages/{page_id}", headers=HEADERS, json=payload)

    if response.status_code == 200:
        log_change("QUARANTINE", page_id, current_title, f"Moved to quarantine as '{new_title}'")
        return True
    else:
        print(f"ERROR quarantining page {page_id}: {response.status_code} - {response.text}")
        return False

def generate_report():
    """Generate detailed change report"""
    report_path = "notion_changes_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("NOTION WORKSPACE REORGANIZATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("SUMMARY OF CHANGES:\n")
        f.write(f"  Total operations: {len(changes_log)}\n")
        f.write(f"  Pages created: {sum(1 for c in changes_log if c['action'] == 'CREATE')}\n")
        f.write(f"  Pages moved: {sum(1 for c in changes_log if c['action'] == 'MOVE')}\n")
        f.write(f"  Pages quarantined: {sum(1 for c in changes_log if c['action'] == 'QUARANTINE')}\n")
        f.write("\n" + "-" * 80 + "\n\n")

        f.write("DETAILED CHANGE LOG:\n\n")
        for change in changes_log:
            f.write(f"[{change['timestamp']}] {change['action']}\n")
            f.write(f"  Page: {change['page_title']}\n")
            f.write(f"  ID: {change['page_id']}\n")
            f.write(f"  Details: {change['details']}\n\n")

        f.write("=" * 80 + "\n")
        f.write("REORGANIZATION COMPLETE\n")
        f.write("Please review the quarantined pages before deleting them.\n")
        f.write("=" * 80 + "\n")

    print(f"\n✅ Report saved to: {report_path}")

def main():
    print("=" * 80)
    print("NOTION WORKSPACE REORGANIZATION")
    print("=" * 80)
    print()

    # Step 1: Create new hubs under Command Center
    print("STEP 1: Creating new hubs under Command Center...\n")
    career_hub = create_page("🎯 Career & Professional Development", "🎯", COMMAND_CENTER)
    daily_ops_hub = create_page("🏠 Daily Operations", "🏠", COMMAND_CENTER)
    quarantine_zone = create_page("🔧 QUARANTINE - Review Before Deleting", "🔧", COMMAND_CENTER)

    if not all([career_hub, daily_ops_hub, quarantine_zone]):
        print("\n❌ ERROR: Failed to create new hubs. Aborting.")
        return

    print()

    # Step 2: Fix orphaned Code Review Agent pages
    print("STEP 2: Restoring orphaned Code Review Agent pages...\n")
    for page_id in ORPHANED_CODE_REVIEW_PAGES:
        move_page(page_id, CODE_REVIEW_AGENT)

    print()

    # Step 3: Quarantine duplicate
    print("STEP 3: Quarantining duplicate page...\n")
    quarantine_page(DUPLICATE_TO_QUARANTINE, quarantine_zone)

    print()

    # Step 4: Generate report
    print("STEP 4: Generating change report...\n")
    generate_report()

    print()
    print("=" * 80)
    print("✅ REORGANIZATION COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Review 'notion_changes_report.txt' for all changes")
    print("  2. Check the '🔧 QUARANTINE' page in Notion")
    print("  3. Manually verify duplicates before deleting")
    print("  4. Organize remaining lifestyle pages into new hubs")
    print()

if __name__ == "__main__":
    main()
