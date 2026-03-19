#!/usr/bin/env python3
"""
Test script to see random descriptions for each skin concern.
Run this to preview how descriptions will appear to different users.
"""

from src.concern_descriptions import get_random_description, get_all_descriptions_for_concerns

def test_single_concern():
    """Test getting random descriptions for a single concern multiple times."""
    print("=" * 80)
    print("Testing: Getting 3 random descriptions for 'Dark Circle'")
    print("=" * 80)
    print()
    
    for i in range(3):
        desc = get_random_description("Dark Circle")
        print(f"--- Random Description #{i+1} ---")
        print(f"Title: {desc['title']}")
        print(f"Description: {desc['description']}")
        print(f"Tips: {desc['tips']}")
        print()


def test_multiple_concerns():
    """Test getting descriptions for multiple concerns at once."""
    print("=" * 80)
    print("Testing: Getting descriptions for multiple concerns (like a real user)")
    print("=" * 80)
    print()
    
    # Simulate a user with multiple skin concerns
    user_concerns = ["PIH", "wrinkle", "blackhead"]
    descriptions = get_all_descriptions_for_concerns(user_concerns)
    
    for label, desc in descriptions.items():
        print(f"🔍 {desc['title']} ({label})")
        print(f"   📝 {desc['description']}")
        print(f"   💡 {desc['tips']}")
        print()


def test_all_concerns():
    """Test all available concerns."""
    print("=" * 80)
    print("Testing: All available skin concerns")
    print("=" * 80)
    print()
    
    all_concerns = ["Dark Circle", "PIH", "blackhead", "whitehead", "papule", "pustule", "wrinkle"]
    
    for concern in all_concerns:
        desc = get_random_description(concern)
        print(f"✨ {concern}")
        print(f"   Title: {desc['title']}")
        print(f"   Description: {desc['description'][:80]}...")
        print(f"   Tips: {desc['tips'][:80]}...")
        print()


def simulate_user_experience():
    """Simulate what 3 different users would see."""
    print("=" * 80)
    print("Simulating: 3 different users with the same concerns")
    print("=" * 80)
    print()
    
    concerns = ["PIH", "wrinkle"]
    
    for user_num in range(1, 4):
        print(f"👤 User {user_num}:")
        descriptions = get_all_descriptions_for_concerns(concerns)
        
        for label, desc in descriptions.items():
            print(f"   • {desc['title']}: {desc['description'][:60]}...")
        print()


if __name__ == "__main__":
    print("\n🧪 Skin Concern Descriptions - Test Suite\n")
    
    # Run all tests
    test_single_concern()
    print("\n" + "=" * 80 + "\n")
    
    test_multiple_concerns()
    print("\n" + "=" * 80 + "\n")
    
    test_all_concerns()
    print("\n" + "=" * 80 + "\n")
    
    simulate_user_experience()
    
    print("\n✅ All tests completed!")
    print("\n💡 Tip: Run this script multiple times to see different random descriptions")
