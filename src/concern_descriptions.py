# src/concern_descriptions.py
"""
Multiple wording variations for each skin concern.
Each user will get a random description to make the experience more personalized.
"""

import random
from typing import Dict, List

# คำอธิบายหลายแบบสำหรับแต่ละปัญหาผิว
CONCERN_DESCRIPTIONS = {
    "Dark Circle": [
        {
            "title": "ถุงใต้ตา",
            "description": "พบรอยคล้ำใต้ดวงตาของคุณ ซึ่งอาจเกิดจากการนอนหลับไม่เพียงพอ ความเครียด หรือพันธุกรรม",
            "tips": "แนะนำให้พักผ่อนให้เพียงพอ ดื่มน้ำมากๆ และใช้ผลิตภัณฑ์บำรุงรอบดวงตาที่มีส่วนผสมของ Vitamin K หรือ Caffeine"
        },
        {
            "title": "รอยคล้ำใต้ตา",
            "description": "ตรวจพบความคล้ำรอบดวงตา ซึ่งเป็นปัญหาที่พบได้บ่อยจากการใช้สายตามากเกินไปหรือการไหลเวียนเลือดไม่ดี",
            "tips": "ลองใช้ Eye Cream ที่มี Retinol หรือ Peptides เพื่อช่วยลดรอยคล้ำ และอย่าลืมนวดเบาๆ รอบดวงตาเพื่อกระตุ้นการไหลเวียน"
        },
        {
            "title": "วงดำรอบดวงตา",
            "description": "คุณมีวงดำรอบดวงตาที่อาจทำให้ดูเหนื่อยล้า ซึ่งอาจเกิดจากอายุที่เพิ่มขึ้นหรือการดูแลไม่เพียงพอ",
            "tips": "ใช้ผลิตภัณฑ์ที่มี Vitamin C และ Niacinamide เพื่อช่วยปรับสีผิวให้สม่ำเสมอ พร้อมทั้งป้องกันแสงแดดด้วย Sunscreen"
        }
    ],
    
    "PIH": [
        {
            "title": "รอยดำจากสิว",
            "description": "พบรอยดำหรือจุดด่างดำบนใบหน้า ซึ่งมักเกิดจากการอักเสบของสิวที่หายไปแล้ว",
            "tips": "ใช้ผลิตภัณฑ์ที่มี Vitamin C, Niacinamide หรือ Alpha Arbutin เพื่อช่วยลดเลือนรอยดำ และป้องกันแสงแดดทุกวัน"
        },
        {
            "title": "จุดด่างดำ",
            "description": "ตรวจพบจุดด่างดำบนผิวหน้า ซึ่งเป็นผลจากการสะสมของเม็ดสีเมลานินมากเกินไป",
            "tips": "แนะนำให้ใช้ Serum ที่มี Tranexamic Acid หรือ Kojic Acid เพื่อช่วยยับยั้งการสร้างเม็ดสี พร้อมทาครีมกันแดด SPF 50+ ทุกวัน"
        },
        {
            "title": "รอยสีผิวไม่สม่ำเสมอ",
            "description": "ผิวของคุณมีสีไม่สม่ำเสมอ มีจุดดำจุดด่างกระจายอยู่ ซึ่งอาจเกิดจากแสงแดดหรือการอักเสบ",
            "tips": "ลองใช้ผลิตภัณฑ์ที่มี AHA/BHA เพื่อผลัดเซลล์ผิว และใช้ Brightening Serum ที่มี Licorice Extract หรือ Vitamin C"
        }
    ],
    
    "blackhead": [
        {
            "title": "สิวหัวดำ",
            "description": "พบสิวหัวดำบริเวณรูขุมขน โดยเฉพาะบริเวณจมูกและคาง ซึ่งเกิดจากการอุดตันของความมันและเซลล์ผิวที่ตาย",
            "tips": "ใช้ผลิตภัณฑ์ที่มี Salicylic Acid หรือ BHA เพื่อช่วยละลายความมันในรูขุมขน และทำความสะอาดผิวหน้าให้ถูกวิธี"
        },
        {
            "title": "รูขุมขนอุดตัน",
            "description": "ตรวจพบรูขุมขนที่อุดตันด้วยความมันและสิ่งสกปรก ทำให้เกิดจุดดำเล็กๆ บนผิวหน้า",
            "tips": "แนะนำให้ใช้ Clay Mask สัปดาห์ละ 1-2 ครั้ง และใช้ Toner ที่มี Witch Hazel หรือ Tea Tree Oil เพื่อควบคุมความมัน"
        },
        {
            "title": "ความมันส่วนเกิน",
            "description": "ผิวของคุณมีความมันมาก ทำให้รูขุมขนอุดตันง่ายและเกิดสิวหัวดำได้บ่อย",
            "tips": "ใช้ผลิตภัณฑ์ Oil-free และ Non-comedogenic พร้อมทั้งใช้ Exfoliating Toner ที่มี Glycolic Acid หรือ Lactic Acid"
        }
    ],
    
    "whitehead": [
        {
            "title": "สิวหัวขาว",
            "description": "พบสิวหัวขาวบนใบหน้า ซึ่งเป็นสิวที่ปิดอยู่ใต้ผิว เกิดจากการอุดตันของรูขุมขนที่ไม่มีการเปิดสู่ภายนอก",
            "tips": "ใช้ผลิตภัณฑ์ที่มี Retinol หรือ Adapalene เพื่อช่วยผลัดเซลล์ผิวและป้องกันการอุดตัน อย่าบีบหรือแคะสิว"
        },
        {
            "title": "สิวอุดตัน",
            "description": "ตรวจพบสิวเล็กๆ สีขาวที่อยู่ใต้ผิว ซึ่งเกิดจากความมันและเซลล์ผิวที่ตายอุดตันในรูขุมขน",
            "tips": "แนะนำให้ใช้ Gentle Exfoliator ที่มี AHA เพื่อช่วยผลัดเซลล์ผิว และใช้ Moisturizer ที่เบาบางไม่อุดตันรูขุมขน"
        },
        {
            "title": "รูขุมขนปิด",
            "description": "ผิวของคุณมีรูขุมขนที่ปิดอุดตัน ทำให้เกิดสิวหัวขาวกระจายอยู่ตามใบหน้า",
            "tips": "ลองใช้ Serum ที่มี Niacinamide เพื่อช่วยควบคุมความมัน และใช้ Cleanser ที่มี Salicylic Acid เพื่อทำความสะอาดรูขุมขนอย่างล้ำลึก"
        }
    ],
    
    "papule": [
        {
            "title": "สิวอักเสบ",
            "description": "พบสิวอักเสบสีแดงบนใบหน้า ซึ่งเป็นสิวที่มีการอักเสบแต่ยังไม่มีหนอง อาจเจ็บเมื่อสัมผัส",
            "tips": "ใช้ผลิตภัณฑ์ที่มี Benzoyl Peroxide หรือ Tea Tree Oil เพื่อลดการอักเสบ และหลีกเลี่ยงการแคะหรือบีบสิว"
        },
        {
            "title": "สิวแดง",
            "description": "ตรวจพบสิวสีแดงที่บวมและอักเสบ ซึ่งเกิดจากแบคทีเรียและการอุดตันของรูขุมขน",
            "tips": "แนะนำให้ใช้ Spot Treatment ที่มี Sulfur หรือ Azelaic Acid เพื่อลดการอักเสบ และรักษาความสะอาดของผิวหน้า"
        },
        {
            "title": "ผิวอักเสบ",
            "description": "ผิวของคุณมีการอักเสบและมีสิวแดงๆ กระจายอยู่ ซึ่งอาจเกิดจากผิวแพ้ง่ายหรือการระคายเคือง",
            "tips": "ใช้ผลิตภัณฑ์ที่อ่อนโยนและไม่มีน้ำหอม พร้อมทั้งใช้ Calming Serum ที่มี Centella Asiatica หรือ Aloe Vera"
        }
    ],
    
    "pustule": [
        {
            "title": "สิวหนอง",
            "description": "พบสิวที่มีหนองสีขาวหรือเหลืองอยู่ตรงกลาง ซึ่งเป็นสิวที่อักเสบรุนแรงและมีการติดเชื้อ",
            "tips": "ใช้ผลิตภัณฑ์ที่มี Benzoyl Peroxide หรือ Salicylic Acid เพื่อฆ่าเชื้อแบคทีเรีย อย่าบีบสิวเพราะอาจทำให้แผลเป็น"
        },
        {
            "title": "สิวติดเชื้อ",
            "description": "ตรวจพบสิวที่มีการติดเชื้อและมีหนองภายใน ซึ่งต้องการการดูแลเป็นพิเศษเพื่อป้องกันแผลเป็น",
            "tips": "แนะนำให้ใช้ Antibiotic Cream หรือ Spot Treatment ที่มี Sulfur พร้อมทั้งรักษาความสะอาดและหลีกเลี่ยงการสัมผัสสิว"
        },
        {
            "title": "สิวรุนแรง",
            "description": "ผิวของคุณมีสิวที่อักเสบรุนแรงและมีหนอง ซึ่งอาจต้องปรึกษาแพทย์ผิวหนังหากไม่ดีขึ้น",
            "tips": "ใช้ผลิตภัณฑ์ที่มี Niacinamide เพื่อลดการอักเสบ และพิจารณาใช้ Retinoid ภายใต้คำแนะนำของแพทย์"
        }
    ],
    
    "wrinkle": [
        {
            "title": "ริ้วรอย",
            "description": "พบริ้วรอยบนใบหน้า โดยเฉพาะบริเวณหน้าผากและรอบดวงตา ซึ่งเป็นสัญญาณของการเสื่อมสภาพของผิวตามอายุ",
            "tips": "ใช้ผลิตภัณฑ์ที่มี Retinol, Peptides หรือ Hyaluronic Acid เพื่อช่วยลดเลือนริ้วรอยและเพิ่มความยืดหยุ่นของผิว"
        },
        {
            "title": "เส้นริ้วรอยแห่งวัย",
            "description": "ตรวจพบเส้นริ้วรอยที่เกิดจากการเคลื่อนไหวของกล้ามเนื้อใบหน้าและการสูญเสียคอลลาเจน",
            "tips": "แนะนำให้ใช้ Anti-aging Serum ที่มี Vitamin C และ Ferulic Acid พร้อมทั้งทาครีมกันแดดทุกวันเพื่อป้องกันริ้วรอยเพิ่มเติม"
        },
        {
            "title": "ผิวเริ่มหย่อนคล้อย",
            "description": "ผิวของคุณเริ่มมีริ้วรอยและความยืดหยุ่นลดลง ซึ่งเป็นเรื่องปกติตามอายุที่เพิ่มขึ้น",
            "tips": "ลองใช้ผลิตภัณฑ์ที่มี Bakuchiol (ทางเลือกแทน Retinol ที่อ่อนโยน) หรือ Adenosine เพื่อช่วยกระตุ้นการสร้างคอลลาเจน"
        }
    ]
}


def get_random_description(concern_label: str) -> Dict[str, str]:
    """
    Get a random description for a given skin concern label.
    
    Args:
        concern_label: Label name (e.g., "Dark Circle", "PIH", "wrinkle")
    
    Returns:
        Dictionary with title, description, and tips
    """
    descriptions = CONCERN_DESCRIPTIONS.get(concern_label, [])
    
    if not descriptions:
        # Fallback if label not found
        return {
            "title": concern_label,
            "description": f"ตรวจพบปัญหา {concern_label} บนใบหน้าของคุณ",
            "tips": "แนะนำให้ปรึกษาผู้เชี่ยวชาญด้านผิวหนังเพื่อรับคำแนะนำที่เหมาะสม"
        }
    
    return random.choice(descriptions)


def get_all_descriptions_for_concerns(concern_labels: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Get random descriptions for multiple skin concerns.
    
    Args:
        concern_labels: List of concern labels
    
    Returns:
        Dictionary mapping label to its random description
    """
    return {
        label: get_random_description(label)
        for label in concern_labels
    }


# For testing
if __name__ == "__main__":
    # Test all concerns
    all_concerns = ["Dark Circle", "PIH", "blackhead", "whitehead", "papule", "pustule", "wrinkle"]
    
    print("=== Testing Random Descriptions ===\n")
    for concern in all_concerns:
        desc = get_random_description(concern)
        print(f"Concern: {concern}")
        print(f"Title: {desc['title']}")
        print(f"Description: {desc['description']}")
        print(f"Tips: {desc['tips']}")
        print("-" * 80)
        print()
