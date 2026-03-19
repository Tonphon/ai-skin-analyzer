# 🔬 AI Skin Analyzer + Product Recommender

ระบบวิเคราะห์ปัญหาผิวหน้าด้วย AI และแนะนำผลิตภัณฑ์สกินแคร์ที่เหมาะสม

## ✨ Features

### 🎯 Skin Concern Detection
- ตรวจจับปัญหาผิวหน้า 7 ประเภทด้วย Deep Learning (EfficientNet V2-S)
- ถ่ายภาพ 3 มุม (หน้าตรง, ซ้าย, ขวา) เพื่อความแม่นยำ
- ตรวจสอบคุณภาพภาพแบบเรียลไทม์ด้วย MediaPipe Face Mesh

### 📝 Personalized Descriptions (NEW!)
- **คำอธิบายแบบสุ่ม**: แต่ละปัญหาผิวมีคำอธิบาย 3 แบบ
- **ประสบการณ์ที่หลากหลาย**: User แต่ละคนได้รับคำอธิบายที่แตกต่างกัน
- **คำแนะนำเฉพาะบุคคล**: แนะนำวิธีดูแลและผลิตภัณฑ์ที่เหมาะสม

### 🛍️ Product Recommendation
- **Cold Start Solution**: แนะนำสินค้าสำหรับลูกค้าใหม่ด้วย Cohort-based CF
- **Personalized Recommendations**: แนะนำตามประวัติการซื้อและข้อมูลประชากรศาสตร์
- **Multiple Strategies**: แนะนำตามปัญหาผิว + แนะนำตามความคล้ายคลึงของ User

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd ai-skin-analyzer

# Install dependencies
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

### Test Descriptions Feature

```bash
# Run test suite
python test_descriptions.py

# Run interactive demo
python demo_descriptions.py

# Interactive mode
python demo_descriptions.py --interactive
```

## 📚 Documentation

- [Concern Descriptions Guide](CONCERN_DESCRIPTIONS.md) - คู่มือการใช้งานคำอธิบายปัญหาผิว
- [API Documentation](docs/API.md) - (Coming soon)
- [User Guide](docs/USER_GUIDE.md) - (Coming soon)

## 🎨 Skin Concerns Detected

| Concern | Thai Name | Variations |
|---------|-----------|------------|
| Dark Circle | ถุงใต้ตา | 3 แบบ |
| PIH | รอยดำจากสิว | 3 แบบ |
| Blackhead | สิวหัวดำ | 3 แบบ |
| Whitehead | สิวหัวขาว | 3 แบบ |
| Papule | สิวอักเสบ | 3 แบบ |
| Pustule | สิวหนอง | 3 แบบ |
| Wrinkle | ริ้วรอย | 3 แบบ |

## 🏗️ Project Structure

```
ai-skin-analyzer/
├── app.py                          # Main Streamlit application
├── src/
│   ├── classifier.py               # Skin concern classification
│   ├── recommender.py              # Product recommendation system
│   ├── config.py                   # Configuration and constants
│   └── concern_descriptions.py     # Random descriptions (NEW!)
├── models/
│   ├── best_model.pth             # Trained classification model
│   └── labels.json                # Class labels
├── data/
│   ├── item_master_with_skin_concern_cat.csv
│   └── sales_fact_skincare_user_features_encrypted_item_number.csv
├── artifacts/
│   └── cf_bundle.pkl              # Pre-computed CF data
├── test_descriptions.py           # Test suite for descriptions
├── demo_descriptions.py           # Interactive demo
└── CONCERN_DESCRIPTIONS.md        # Documentation

```

## 💡 Usage Examples

### Basic Usage

```python
from src.classifier import SkinConcernClassifier
from src.concern_descriptions import get_all_descriptions_for_concerns

# Load classifier
clf = SkinConcernClassifier("models/best_model.pth")

# Classify image
prediction = clf.predict("path/to/image.jpg")

# Get random descriptions
descriptions = get_all_descriptions_for_concerns(prediction.positive_labels)

# Display results
for label in prediction.positive_labels:
    desc = descriptions[label]
    print(f"{desc['title']}: {desc['description']}")
```

### Advanced Usage

```python
from src.recommender import Recommender

# Load recommender
rec = Recommender("artifacts/cf_bundle.pkl")

# For new user
recommendations = rec.recommend_new_user(
    gender="F",
    birth_year=1997,
    selected_concern_ids=[1, 2],  # Whitening + Anti-Aging
    top_k=10
)

# For existing user
recommendations = rec.recommend_by_concern(
    user_id="USER123",
    selected_concern_ids=[1, 2],
    top_k=10,
    allow_repeats=True,
    repurchase_boost=1.2
)
```

## 🔧 Technologies

- **Deep Learning**: PyTorch, EfficientNet V2-S
- **Computer Vision**: MediaPipe, OpenCV
- **Recommendation**: Collaborative Filtering, Cosine Similarity
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy

## 📊 Performance

- **Classification Accuracy**: ~85% (7-class multilabel)
- **Recommendation Precision@10**: ~0.24 (existing users)
- **Cold Start NDCG@10**: ~0.58 (new users)

## 🎯 Future Enhancements

- [ ] Multi-language support (English)
- [ ] Severity-based descriptions
- [ ] User preference tracking
- [ ] Analytics dashboard
- [ ] Mobile app version
- [ ] Real-time video analysis

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines first.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Team

- AI/ML Team
- Backend Team
- Frontend Team
- Product Team

## 📞 Contact

For questions or support, please contact: [your-email@example.com]

---

**Version**: 1.0.0  
**Last Updated**: 2025  
**Status**: Active Development