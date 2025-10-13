"""
Минимальный пример OCR инференса - всё в 15 строк.
"""

from inference import OCRInference

def main():
    # Пути
    MODEL_PATH = r"exp_4_model\best_acc_weights.pth"
    CHARSET_PATH = "configs\\charset.txt"
    IMAGE_PATH = r"C:\shared\orig_cyrillic\test\test0.png"
    
    # Загрузка модели и распознавание
    ocr = OCRInference(MODEL_PATH, CHARSET_PATH, device="auto")
    text = ocr.predict(IMAGE_PATH)
    print(f"Результат: '{text}'")

if __name__ == "__main__":
    main()