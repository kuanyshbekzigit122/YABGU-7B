import os
import json
import re

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
from datasets import load_dataset
from tokenizers import Tokenizer

output_dir = "/content/data/sft"
os.makedirs(output_dir, exist_ok=True)
out_file = os.path.join(output_dir, "kazakh_sft_combined.jsonl")

# 1. Custom Curated Kazakh Identity, Culture, History & Dialogue
curated_data = [
    {
        "instruction": "Сен кімсің?",
        "input": "",
        "output": "Мен YABGU (Ябғу) — қазақ тілінде дербес оқытылған жасанды интеллект тілдік моделімін. Менің мақсатым — қазақ тілінде сұрақтарға сапалы жауап беру, мәтіндерді өңдеу және сізге жан-жақты көмек көрсету."
    },
    {
        "instruction": "YABGU атауы нені білдіреді?",
        "input": "",
        "output": "YABGU (Ябғу) — ежелгі Түркі мемлекеттеріндегі (Түрік қағанаты, Оғыз мемлекеті) қағаннан кейінгі ең жоғары билеуші лауазымдарының бірі. Бұл жобада ол қазақ тіліндегі дербес әрі қуатты ұлттық жасанды интеллект моделі ретінде таңдалған."
    },
    {
        "instruction": "Сәлеметсіз бе! Қалың қалай?",
        "input": "",
        "output": "Сәлеметсіз бе! Менде бәрі жақсы, рақмет! Өзіңіздің көңіл-күйіңіз қалай? Сізге бүгін қандай көмек көрсете аламын?"
    },
    {
        "instruction": "Қазақстанның астанасы қандай қала?",
        "input": "",
        "output": "Қазақстанның астанасы — Астана қаласы. Ол Есіл өзенінің жағасында орналасқан және еліміздің басты саяси, мәдени әрі экономикалық орталығы болып табылады."
    },
    {
        "instruction": "Абай Құнанбаев кім және оның қазақ әдебиетіндегі орны қандай?",
        "input": "",
        "output": "Абай (Ибраһим) Құнанбаев (1845–1904) — қазақтың ұлы ақыны, ойшылы, ағартушысы және қазақ жазба әдебиетінің негізін қалаушы. Оның өлеңдері, поэмалары және әйгілі «Қара сөздері» қазақ халқының рухани қазынасы мен философиялық ой-өрісінің биік шыңы болып саналады."
    },
    {
        "instruction": "Алаш Орда үкіметі қашан құрылды және оның жетекшілері кімдер?",
        "input": "",
        "output": "Алаш Орда үкіметі (Алаш автономиясы) 1917 жылы желтоқсан айында Орынборда өткен екінші жалпықазақ съезінде құрылды. Оның төрағасы болып Әлихан Бөкейханов сайланды. Алаш қозғалысының көрнекті қайраткерлері қатарында Ахмет Байтұрсынұлы, Міржақып Дулатұлы, Мағжан Жұмабаев, Халел Досмұхамедов және Жүсіпбек Аймауытов бар."
    },
    {
        "instruction": "Қазақ тілінде қандай септіктер бар? Тізіп беріңіз.",
        "input": "",
        "output": "Қазақ тілінде 7 септік бар:\n1. Атау септік (Кім? Не?)\n2. Ілік септік (Кімнің? Ненің? Жұрнақтары: -ның/-нің, -дың/-дің, -тың/-тің)\n3. Барыс септік (Кімге? Неге? Қайда? Жұрнақтары: -ға/-ге, -қа/-ке, -на/-не)\n4. Табыс септік (Кімді? Нені? Жұрнақтары: -ны/-ні, -ды/-ді, -ты/-ті, -н)\n5. Жатыс септік (Кімде? Неде? Қайда? Жұрнақтары: -да/-де, -та/-те, -нда/-нде)\n6. Шығыс септік (Кімнен? Неден? Қайдан? Жұрнақтары: -нан/-нен, -дан/-ден, -тан/-тен)\n7. Көмектес септік (Кіммен? Немен? Жұрнақтары: -мен, -бен, -пен)."
    },
    {
        "instruction": "Жасанды интеллект дегеніміз не?",
        "input": "",
        "output": "Жасанды интеллект (AI) — адамның ойлау, үйрену, шешім қабылдау және мәселелерді шешу сияқты когнитивті қабілеттерін компьютерлік жүйелер мен алгоритмдер арқылы модельдеуге бағытталған информатиканың саласы."
    },
    {
        "instruction": "Python тілінде тізімді (list) қалай сұрыптауға болады?",
        "input": "",
        "output": "Python тілінде тізімді екі тәсілмен сұрыптауға болады:\n1. Тізімнің өзгерту әдісі `.sort()` арқылы:\n```python\nnumbers = [5, 2, 9, 1]\nnumbers.sort()\nprint(numbers)  # [1, 2, 5, 9]\n```\n2. Жаңа сұрыпталған тізім қайтаратын `sorted()` функциясы арқылы:\n```python\nnumbers = [5, 2, 9, 1]\nnew_list = sorted(numbers)\nprint(new_list)  # [1, 2, 5, 9]\n```"
    },
    {
        "instruction": "Маған достық туралы қазақша мақал-мәтелдер айтып берші.",
        "input": "",
        "output": "Достық туралы танымал қазақ мақал-мәтелдері:\n1. «Жүз теңгең болғанша, жүз досың болсын».\n2. «Дос жылатып айтады, дұшпан күлдіріп айтады».\n3. «Ағаш тамырымен мықты, адам досымен мықты».\n4. «Жаңа дос тапсаң да, ескі досты ұмытпа»."
    },
    {
        "instruction": "Мәтінді қазақ тіліне аударыңыз: 'Machine learning is a subset of artificial intelligence.'",
        "input": "",
        "output": "«Машиналық оқыту — жасанды интеллекттің бір тармағы (ішкі жиыны).»"
    },
    {
        "instruction": "Алматы қаласы туралы қысқаша мәлімет беріңіз.",
        "input": "",
        "output": "Алматы — Қазақстанның ең ірі мегаполисі және мәдени, қаржылық, білім беру орталығы. Ол Іле Алатауының бөктерінде орналасқан, бұрын Қазақстанның астанасы болған. Қала Медеу биік таулы спорт кешені, Шымбұлақ тау-шаңғы курорты және Көк-Төбе демалыс саябағымен танымал."
    }
]

collected_samples = list(curated_data)
print(f"[CURATED] Қолмен әзірленген сапалы үлгілер саны: {len(collected_samples)}")

# 2. AmanMussa/kazakh-instruction-v2 жүктеу
print("[HF] AmanMussa/kazakh-instruction-v2 жүктелуде...")
try:
    ds_aman = load_dataset("AmanMussa/kazakh-instruction-v2", split="train")
    count_aman = 0
    for item in ds_aman:
        inst = item.get("instruction", "").strip()
        inp = item.get("input", "").strip()
        out = item.get("output", "").strip()
        if inst and out and len(out) > 5 and len(inst) > 5:
            collected_samples.append({
                "instruction": inst,
                "input": inp if inp and inp.lower() != "nan" else "",
                "output": out
            })
            count_aman += 1
    print(f"[HF] AmanMussa-дан алынған үлгілер саны: {count_aman:,}")
except Exception as e:
    print(f"[ERR] AmanMussa жүктелмеді: {e}")

# 3. DarkyMan/powerful-kazakh-dialogue диалогтарын өңдеу
print("[HF] DarkyMan/powerful-kazakh-dialogue жүктелуде...")
try:
    ds_dialogue = load_dataset("DarkyMan/powerful-kazakh-dialogue", split="train")
    count_dialogue = 0
    for item in ds_dialogue:
        dialogue = item.get("dialogue", [])
        # Диалогтағы user -> assistant жұптарын алу
        for i in range(len(dialogue) - 1):
            if dialogue[i].get("role") == "user" and dialogue[i+1].get("role") == "assistant":
                u_text = dialogue[i].get("content", "").strip()
                a_text = dialogue[i+1].get("content", "").strip()
                if u_text and a_text and len(u_text) > 5 and len(a_text) > 10:
                    collected_samples.append({
                        "instruction": u_text,
                        "input": "",
                        "output": a_text
                    })
                    count_dialogue += 1
    print(f"[HF] DarkyMan диалогтарынан алынған жұптар саны: {count_dialogue:,}")
except Exception as e:
    print(f"[ERR] DarkyMan жүктелмеді: {e}")

# 4. JSONL форматында сақтау
print(f"\n[SAVE] Жалпы жинақталған үлгілер: {len(collected_samples):,}")
with open(out_file, "w", encoding="utf-8") as f:
    for item in collected_samples:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

file_size_mb = os.path.getsize(out_file) / (1024 * 1024)
print(f"[SUCCESS] SFT деректер файлы сақталды: {out_file} ({file_size_mb:.2f} MB)")
