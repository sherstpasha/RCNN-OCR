from utils import CharNGramLM

# 1) читаем все слова
with open(r"all_words.txt", encoding="utf-8") as f:
    words = [line.strip() for line in f if line.strip()]

# 2) обучаем 6‑граммную модель
lm = CharNGramLM(N=6, alpha=1.0, case_insensitive=True)
lm.train(words)

# 3) сохраняем
lm.save("char6gram.pkl")
