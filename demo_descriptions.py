#!/usr/bin/env python3
"""
Interactive demo to explore skin concern descriptions.
Run this to see how descriptions work in a user-friendly way.
"""

import sys
from src.concern_descriptions import (
    get_random_description, 
    get_all_descriptions_for_concerns,
    CONCERN_DESCRIPTIONS
)

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def print_description(desc, label=None):
    """Print a formatted description."""
    if label:
        print(f"🏷️  Label: {label}")
    print(f"✨ {desc['title']}")
    print(f"\n📝 คำอธิบาย:")
    print(f"   {desc['description']}")
    print(f"\n💡 คำแนะนำ:")
    print(f"   {desc['tips']}")
    print()


def demo_single_concern():
    """Demo: Show all variations for a single concern."""
    print_header("Demo 1: All Variations for a Single Concern")
    
    concern = "Dark Circle"
    print(f"Showing all variations for: {concern}\n")
    
    variations = CONCERN_DESCRIPTIONS[concern]
    for i, desc in enumerate(variations, 1):
        print(f"--- Variation {i} ---")
        print_description(desc)
        print("-" * 80)


def demo_random_selection():
    """Demo: Show random selection multiple times."""
    print_header("Demo 2: Random Selection (3 times)")
    
    concern = "PIH"
    print(f"Getting random descriptions for: {concern}")
    print("(Notice how each call might return a different variation)\n")
    
    for i in range(3):
        print(f"🎲 Random Call #{i+1}:")
        desc = get_random_description(concern)
        print(f"   Title: {desc['title']}")
        print(f"   Description: {desc['description'][:60]}...")
        print()


def demo_multiple_concerns():
    """Demo: Simulate a real user with multiple concerns."""
    print_header("Demo 3: Real User Scenario")
    
    print("👤 User Profile:")
    print("   - Age: 28")
    print("   - Gender: Female")
    print("   - Detected Concerns: PIH, wrinkle, blackhead\n")
    
    concerns = ["PIH", "wrinkle", "blackhead"]
    descriptions = get_all_descriptions_for_concerns(concerns)
    
    print("📊 Analysis Results:\n")
    for label in concerns:
        desc = descriptions[label]
        print_description(desc, label)
        print("-" * 80)


def demo_comparison():
    """Demo: Compare what 3 different users see."""
    print_header("Demo 4: User Experience Comparison")
    
    concern = "wrinkle"
    print(f"3 different users with the same concern: {concern}\n")
    
    for user_num in range(1, 4):
        desc = get_random_description(concern)
        print(f"👤 User {user_num} sees:")
        print(f"   Title: {desc['title']}")
        print(f"   Description: {desc['description'][:70]}...")
        print()


def demo_all_concerns():
    """Demo: Show one random description for each concern."""
    print_header("Demo 5: All Skin Concerns Overview")
    
    all_concerns = ["Dark Circle", "PIH", "blackhead", "whitehead", "papule", "pustule", "wrinkle"]
    
    print("Random description for each concern type:\n")
    
    for concern in all_concerns:
        desc = get_random_description(concern)
        print(f"🔍 {concern}")
        print(f"   ✨ {desc['title']}")
        print(f"   📝 {desc['description'][:65]}...")
        print()


def interactive_mode():
    """Interactive mode: Let user choose concerns."""
    print_header("Interactive Mode")
    
    all_concerns = ["Dark Circle", "PIH", "blackhead", "whitehead", "papule", "pustule", "wrinkle"]
    
    print("Available concerns:")
    for i, concern in enumerate(all_concerns, 1):
        print(f"  {i}. {concern}")
    
    print("\nEnter concern numbers separated by commas (e.g., 1,3,7)")
    print("Or press Enter to see all concerns")
    
    try:
        user_input = input("\nYour choice: ").strip()
        
        if not user_input:
            selected = all_concerns
        else:
            indices = [int(x.strip()) - 1 for x in user_input.split(",")]
            selected = [all_concerns[i] for i in indices if 0 <= i < len(all_concerns)]
        
        if not selected:
            print("\n❌ No valid concerns selected.")
            return
        
        print(f"\n✅ Selected: {', '.join(selected)}\n")
        
        descriptions = get_all_descriptions_for_concerns(selected)
        
        for label in selected:
            desc = descriptions[label]
            print_description(desc, label)
            print("-" * 80)
            
    except (ValueError, IndexError):
        print("\n❌ Invalid input. Please enter numbers separated by commas.")


def main():
    """Main demo runner."""
    print("\n" + "🎨" * 40)
    print("\n  Skin Concern Descriptions - Interactive Demo")
    print("\n" + "🎨" * 40)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
        return
    
    # Run all demos
    demo_single_concern()
    input("\nPress Enter to continue...")
    
    demo_random_selection()
    input("\nPress Enter to continue...")
    
    demo_multiple_concerns()
    input("\nPress Enter to continue...")
    
    demo_comparison()
    input("\nPress Enter to continue...")
    
    demo_all_concerns()
    
    print("\n" + "=" * 80)
    print("\n✅ Demo completed!")
    print("\n💡 Tips:")
    print("   - Run this script multiple times to see different random results")
    print("   - Use --interactive flag for interactive mode:")
    print("     python demo_descriptions.py --interactive")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
