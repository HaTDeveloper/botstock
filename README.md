# بوت تحليل الأسهم السعودية
# Saudi Stock Analysis Bot

## نظرة عامة | Overview

بوت تحليل الأسهم السعودية هو أداة مصممة لتحليل بيانات سوق الأسهم السعودية وتحديد فرص الاستثمار المحتملة. يقدم البوت نوعين من التحليل:

1. **الفرص الذهبية (ارتفاعات وانخفاضات سريعة)**: تحديد الأسهم التي يُتوقع أن ترتفع أو تنخفض بسرعة في المدى القصير (1-5 أيام تداول).
2. **تحليل عام للأسهم المتوقع ارتفاعها**: تحديد الأسهم ذات إمكانية النمو على المدى المتوسط (1-3 أشهر).

The Saudi Stock Analysis Bot is a tool designed to analyze Saudi stock market data and identify potential investment opportunities. The bot provides two types of analysis:

1. **Golden Opportunities (Quick Rises and Falls)**: Identifying stocks expected to rise or fall quickly in the short term (1-5 trading days).
2. **General Analysis of Stocks Expected to Rise**: Identifying stocks with growth potential in the medium term (1-3 months).

## الميزات | Features

- تحليل البيانات في الوقت الفعلي عند توفرها
- تحليل البيانات التاريخية (حتى 7 سنوات عند توفرها)
- دمج تحليل الأخبار وتأثيرها
- مقاييس الثقة للتوقعات
- نصائح استثمارية مع كل توصية
- تحديد أرقام الأسهم لسهولة العثور عليها
- تقارير باللغتين العربية والإنجليزية

- Real-time data analysis when available
- Historical data analysis (up to 7 years when available)
- News integration and impact analysis
- Confidence metrics for predictions
- Investment advice with each recommendation
- Stock numbers for easy identification
- Reports in both Arabic and English

## متطلبات النظام | System Requirements

- Python 3.6+
- المكتبات: pandas, numpy, matplotlib
- اتصال بالإنترنت للوصول إلى واجهات برمجة التطبيقات (APIs)

- Python 3.6+
- Libraries: pandas, numpy, matplotlib
- Internet connection for API access

## التثبيت | Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/saudi-stock-bot.git
cd saudi-stock-bot

# Install required packages
pip install pandas numpy matplotlib
```

## الاستخدام | Usage

### النسخة الأساسية | Basic Version

```bash
python src/saudi_stock_bot.py
```

### النسخة المحسنة | Enhanced Version

```bash
python src/enhanced_saudi_stock_bot.py
```

## تخصيص البوت | Customization

يمكن تخصيص البوت من خلال تعديل المعلمات التالية في الكود:

1. **عتبات الثقة**: يمكن تعديل عتبات الثقة العالية والمتوسطة والمنخفضة لتغيير حساسية البوت في اكتشاف الفرص.

```python
# Confidence thresholds
self.high_confidence = 70  # كان 80 سابقاً
self.medium_confidence = 50  # كان 60 سابقاً
self.low_confidence = 30  # كان 40 سابقاً
```

2. **قائمة الأسهم**: يمكن تعديل قائمة الأسهم السعودية التي يتم تحليلها في دالة `load_saudi_market_symbols`.

3. **عدد الأسهم للتحليل**: يمكن تغيير عدد الأسهم التي يتم تحليلها في دالة `analyze_stocks`.

```python
analysis_results = bot.analyze_stocks(analysis_type="both", max_stocks=15)
```

4. **نوع التحليل**: يمكن تحديد نوع التحليل المطلوب ("golden" للفرص الذهبية، "general" للتحليل العام، أو "both" لكليهما).

5. **لغة التقرير**: يمكن تغيير لغة التقرير إلى العربية ("ar") أو الإنجليزية ("en").

```python
report = bot.generate_report(analysis_results, lang="ar")
```

The bot can be customized by modifying the following parameters in the code:

1. **Confidence Thresholds**: The high, medium, and low confidence thresholds can be adjusted to change the bot's sensitivity in detecting opportunities.

```python
# Confidence thresholds
self.high_confidence = 70  # Was 80
self.medium_confidence = 50  # Was 60
self.low_confidence = 30  # Was 40
```

2. **Stock List**: The list of Saudi stocks to be analyzed can be modified in the `load_saudi_market_symbols` function.

3. **Number of Stocks to Analyze**: The number of stocks to analyze can be changed in the `analyze_stocks` function.

```python
analysis_results = bot.analyze_stocks(analysis_type="both", max_stocks=15)
```

4. **Analysis Type**: The type of analysis can be specified ("golden" for golden opportunities, "general" for general analysis, or "both" for both).

5. **Report Language**: The report language can be changed to Arabic ("ar") or English ("en").

```python
report = bot.generate_report(analysis_results, lang="ar")
```

## تفسير النتائج | Interpreting Results

### نسبة الثقة | Confidence Percentage

- **70-100%**: ثقة عالية في التوقع
- **50-69%**: ثقة متوسطة في التوقع
- **30-49%**: ثقة منخفضة في التوقع

- **70-100%**: High confidence in prediction
- **50-69%**: Medium confidence in prediction
- **30-49%**: Low confidence in prediction

### توصيات الاستثمار | Investment Recommendations

- **شراء قوي (Strong Buy)**: الأسهم ذات مؤشرات قوية للنمو على المدى المتوسط
- **شراء (Buy)**: الأسهم ذات إمكانية جيدة للنمو على المدى المتوسط
- **مراقبة (Watch)**: الأسهم ذات بعض الإمكانات ولكن تتطلب المراقبة
- **احتفاظ/تجنب (Hold/Avoid)**: الأسهم التي لا تظهر إمكانية نمو قوية في الوقت الحالي

- **Strong Buy**: Stocks with strong indicators for medium-term growth
- **Buy**: Stocks with good potential for medium-term growth
- **Watch**: Stocks with some potential but requiring monitoring
- **Hold/Avoid**: Stocks that do not show strong growth potential at this time

## الفرق بين النسختين | Difference Between Versions

### النسخة الأساسية (saudi_stock_bot.py) | Basic Version

- تحليل 15 سهماً سعودياً
- عتبات ثقة أعلى (80/60/40)
- مؤشرات فنية أساسية

- Analyzes 15 Saudi stocks
- Higher confidence thresholds (80/60/40)
- Basic technical indicators

### النسخة المحسنة (enhanced_saudi_stock_bot.py) | Enhanced Version

- تحليل 38 سهماً سعودياً
- عتبات ثقة أقل للحساسية العالية (70/50/30)
- مؤشرات فنية إضافية (ADX، Stochastic Oscillator، OBV، Ichimoku Cloud)
- خوارزميات تحليل محسنة مع شروط أكثر دقة
- دمج معلومات تداول المطلعين (insider trading)

- Analyzes 38 Saudi stocks
- Lower confidence thresholds for higher sensitivity (70/50/30)
- Additional technical indicators (ADX, Stochastic Oscillator, OBV, Ichimoku Cloud)
- Improved analysis algorithms with more nuanced conditions
- Integration of insider trading information

## ملاحظات هامة | Important Notes

- هذا التحليل لأغراض إعلامية فقط ولا ينبغي اعتباره نصيحة مالية.
- قم دائمًا بإجراء البحث الخاص بك وفكر في استشارة مستشار مالي قبل اتخاذ قرارات الاستثمار.
- يعتمد البوت على واجهات برمجة تطبيقات Yahoo Finance، والتي قد تكون محدودة في بعض الأحيان للأسهم السعودية.
- قد لا يكتشف البوت فرصاً في جميع ظروف السوق، وهذا لا يعني بالضرورة وجود خلل في البوت.

- This analysis is for informational purposes only and should not be considered as financial advice.
- Always conduct your own research and consider consulting with a financial advisor before making investment decisions.
- The bot relies on Yahoo Finance APIs, which may be limited at times for Saudi stocks.
- The bot may not detect opportunities in all market conditions, which does not necessarily indicate a flaw in the bot.

## المساهمة | Contributing

المساهمات مرحب بها! يرجى إرسال طلبات السحب (pull requests) أو فتح مشكلات (issues) للتحسينات المقترحة.

Contributions are welcome! Please submit pull requests or open issues for suggested improvements.

## الترخيص | License

[MIT License](LICENSE)
"# saudi-stock-bot" 
"# saudi-stock-bot" 
