# FruitLens AI 🍎

FruitLens AI is a Streamlit computer-vision school project that detects fruit varieties in uploaded images, displays bounding boxes and confidence scores, counts detected fruit instances, and provides approximate nutrition reference values.

## Fruit labels configured

The YOLO-World open-vocabulary detector is prompted with 45 fruit labels: apple, banana, orange, mango, grapes, watermelon, strawberry, pineapple, papaya, pear, peach, plum, cherry, kiwi, pomegranate, guava, coconut, lemon, lime, avocado, blueberry, raspberry, blackberry, cranberry, dragon fruit, passion fruit, lychee, rambutan, jackfruit, durian, fig, date, apricot, nectarine, grapefruit, tangerine, mandarin, custard apple, sapodilla, star fruit, mulberry, Indian gooseberry (amla), jamun, longan, and persimmon.

## Run locally

```bash
pip install -r requirements.txt
streamlit run foodlens_app.py
```

## Deploy on Streamlit Community Cloud

1. Upload the contents of this folder to the root of your GitHub repository.
2. In Streamlit Community Cloud, select `foodlens_app.py` as the Main file path.
3. Deploy (or reboot the existing app after pushing the commit).

The repository keeps the existing `packages.txt` system dependencies and Python 3.11 runtime configuration.

## Important model note

This project uses the pretrained YOLO-World open-vocabulary model and configures its text prompts for 45 fruit categories. It is **not a newly trained custom neural-network checkpoint**. Prompting the model with a label makes it a requested detection category, but does not guarantee that every variety or every image will be recognized. True custom training would require a labeled fruit-image dataset, train/validation splits, and a separate training run.

## Nutrition note

Nutrition figures are approximate reference values per 100 g. The app does not measure fruit weight or variety, so calculated totals are educational estimates rather than exact dietary information.
