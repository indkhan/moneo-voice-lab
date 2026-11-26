p='evallib.py'
s=open(p,encoding='utf-8').read()
a=s.index('def save_result(section, key, value):'); b=s.index('def metrics(gold, pred):')
new='''def save_result(section, key, value):
    """Several experiment processes write at once: keep one file per entry (the source of truth) and rebuild results.json under a lock."""
    parts = os.path.join(B, "results_parts"); os.makedirs(parts, exist_ok=True)
    with open(os.path.join(parts, f"{section}__{key}.json"), "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False)
    lock = RESULTS + ".lock"
    for _ in range(200):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd); break
        except FileExistsError:
            time.sleep(0.05)
    try:
        data = json.load(open(RESULTS, encoding="utf-8")) if os.path.exists(RESULTS) else {}
        for name in sorted(os.listdir(parts)):
            sec, k = name[:-5].split("__", 1)
            data.setdefault(sec, {})[k] = json.load(open(os.path.join(parts, name), encoding="utf-8"))
        with open(RESULTS, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
    finally:
        try: os.remove(lock)
        except OSError: pass


'''
open(p,'w',encoding='utf-8').write(s[:a]+new+s[b:])
