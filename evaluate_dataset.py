"""
Скрипт для вычисления метрик OCR модели на датасете.

Использование:
    python evaluate_dataset.py --model model.pth --charset charset.txt --csv labels.csv --root images/
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np

from inference import OCRInference
from training.metrics import character_error_rate, word_error_rate, compute_accuracy


def load_dataset(csv_path, root_path):
    """Загружает датасет из CSV файла."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV файл не найден: {csv_path}")
    
    if not os.path.exists(root_path):
        raise FileNotFoundError(f"Папка с изображениями не найдена: {root_path}")
    
    # Читаем CSV
    df = pd.read_csv(csv_path)
    
    # Ожидаем колонки: filename, text
    if 'filename' not in df.columns or 'text' not in df.columns:
        raise ValueError("CSV должен содержать колонки 'filename' и 'text'")
    
    # Формируем полные пути к изображениям
    image_paths = []
    texts = []
    
    for idx, row in df.iterrows():
        filename = row['filename']
        text = str(row['text'])
        
        # Ищем файл в папке
        image_path = os.path.join(root_path, filename)
        
        # Если нет расширения, пробуем разные
        if not os.path.exists(image_path):
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                test_path = os.path.join(root_path, filename + ext)
                if os.path.exists(test_path):
                    image_path = test_path
                    break
        
        if os.path.exists(image_path):
            image_paths.append(image_path)
            texts.append(text)
        else:
            print(f"⚠️  Изображение не найдено: {filename}")
    
    return image_paths, texts


def evaluate_model(model_path, charset_path, csv_path, root_path, batch_size=16, max_samples=None, img_h=32, img_w=128):
    """Оценивает модель на датасете."""
    
    print(f"📊 Оценка модели на датасете")
    print(f"📂 Модель: {model_path}")
    print(f"📂 Charset: {charset_path}")
    print(f"📂 CSV: {csv_path}")
    print(f"📂 Изображения: {root_path}")
    print(f"📐 Размеры изображения: {img_h}x{img_w}")
    print("-" * 60)
    
    # Загружаем модель
    print("🔄 Загружаем модель...")
    ocr = OCRInference(model_path, charset_path, device="auto", img_h=img_h, img_w=img_w)
    
    # Загружаем датасет
    print("📁 Загружаем датасет...")
    image_paths, true_texts = load_dataset(csv_path, root_path)
    
    if max_samples:
        image_paths = image_paths[:max_samples]
        true_texts = true_texts[:max_samples]
    
    print(f"📋 Найдено {len(image_paths)} образцов")
    
    if len(image_paths) == 0:
        print("❌ Нет данных для оценки!")
        return
    
    # Предсказания
    print("🚀 Выполняем предсказания...")
    predicted_texts = []
    
    # Батчевые предсказания с прогресс-баром
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Обработка"):
        batch_paths = image_paths[i:i + batch_size]
        batch_predictions = ocr.predict(batch_paths, batch_size=batch_size)
        predicted_texts.extend(batch_predictions)
    
    # Вычисляем метрики
    print("\n📊 Вычисляем метрики...")
    
    # Точность (exact match)
    accuracy = compute_accuracy(true_texts, predicted_texts)
    
    # CER (Character Error Rate)
    cers = [character_error_rate(true, pred) for true, pred in zip(true_texts, predicted_texts)]
    avg_cer = np.mean(cers)
    
    # WER (Word Error Rate) 
    wers = []
    for true, pred in zip(true_texts, predicted_texts):
        try:
            wer = word_error_rate(true, pred)
            wers.append(wer)
        except:
            # В случае ошибки (например, пустые строки) считаем как 100% ошибку
            wers.append(1.0)
    avg_wer = np.mean(wers)
    
    # Результаты
    print("\n" + "="*60)
    print("📈 РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("="*60)
    print(f"📊 Количество образцов: {len(image_paths)}")
    print(f"🎯 Точность (Exact Match): {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"🔤 Средний CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
    print(f"📝 Средний WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
    
    # Статистика ошибок
    print(f"\n📉 Статистика ошибок:")
    print(f"CER: мин={min(cers):.3f}, макс={max(cers):.3f}, медиана={np.median(cers):.3f}")
    print(f"WER: мин={min(wers):.3f}, макс={max(wers):.3f}, медиана={np.median(wers):.3f}")
    
    # Примеры ошибок
    print(f"\n❌ Примеры ошибок (топ-5 по CER):")
    error_data = list(zip(true_texts, predicted_texts, cers))
    error_data.sort(key=lambda x: x[2], reverse=True)
    
    for i, (true, pred, cer) in enumerate(error_data[:5]):
        print(f"{i+1}. CER={cer:.3f}")
        print(f"   Истинный: '{true}'")
        print(f"   Предсказанный: '{pred}'")
        print()
    
    # Сохраняем подробные результаты
    results_df = pd.DataFrame({
        'image_path': [os.path.basename(p) for p in image_paths],
        'true_text': true_texts,
        'predicted_text': predicted_texts,
        'cer': cers,
        'wer': wers,
        'exact_match': [t == p for t, p in zip(true_texts, predicted_texts)]
    })
    
    output_path = f"evaluation_results_{os.path.basename(model_path)}.csv"
    results_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"💾 Подробные результаты сохранены в: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Оценка OCR модели на датасете")
    
    parser.add_argument("--model", type=str, required=True, help="Путь к файлу модели (.pth)")
    parser.add_argument("--charset", type=str, required=True, help="Путь к файлу charset")
    parser.add_argument("--csv", type=str, required=True, help="Путь к CSV файлу с разметкой")
    parser.add_argument("--root", type=str, required=True, help="Папка с изображениями")
    parser.add_argument("--batch-size", type=int, default=16, help="Размер батча (default: 16)")
    parser.add_argument("--max-samples", type=int, default=None, help="Максимальное количество образцов для оценки")
    parser.add_argument("--img-h", type=int, default=32, help="Высота входного изображения (default: 32)")
    parser.add_argument("--img-w", type=int, default=128, help="Ширина входного изображения (default: 128)")
    
    args = parser.parse_args()
    
    # Проверяем файлы
    if not os.path.exists(args.model):
        print(f"❌ Модель не найдена: {args.model}")
        return 1
    
    if not os.path.exists(args.charset):
        print(f"❌ Charset не найден: {args.charset}")
        return 1
    
    try:
        evaluate_model(
            model_path=args.model,
            charset_path=args.charset,
            csv_path=args.csv,
            root_path=args.root,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            img_h=args.img_h,
            img_w=args.img_w
        )
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())