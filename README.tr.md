# SFTDiff

**İki eğitim hattının modele verdiği tokenları ve öğrenme hedeflerini karşılaştırır.**

İlk çalışan alfa sürümüdür. Hazır demo sentetiktir; farklı eğitim framework'lerinde kapsamlı doğrulama henüz yapılmadı.

Proje klasöründe:

```powershell
python -m sftdiff demo --out demo-run
```

`demo-run/report.html` dosyasını tarayıcıda aç. Ya da projeyle gelen `demo/report.html` dosyasını doğrudan açabilirsin.

Demo; etiket değişimi, cevabın kesilmesi, EOS kaybı, padding'in öğrenme hedefine katılması ve eklenen/silinen kayıtları gösterir. Bunlar elle hazırlanmış örneklerdir; gerçek bir model eğitiminden alınmış ölçüm değildir.

Gerçek kullanımda `capture_batch` fonksiyonuna kendi collator'ından çıkan gerçek `input_ids`, `labels`, isteğe bağlı `attention_mask` tensorlarını ve aynı sıradaki sabit kayıt kimliklerini verirsin. Araç tokenları yeniden üretmez veya etiketleri değiştirmez. İki koşudan aldığın dosyaları karşılaştırırsın:

```powershell
python -m sftdiff compare before.jsonl after.jsonl --html report.html --json evidence.json
```

Çekirdek araç GPU, model ağırlıkları veya API anahtarı gerektirmez. İlk kapsam; ortak tokenizer sözlüğü, metin tabanlı paketlenmemiş örnekler ve kaydırılmamış causal-LM etiketleridir. Farklılaşma otomatik olarak hata sayılmaz. HTML raporu yereldir; örnek eğitim metnini içerdiği için özel verilerle üretilen raporları paylaşırken dikkat edilmelidir.

Testler:

```powershell
python -m unittest discover -s tests -v
```

Tam kullanım, sınırlar ve mevcut araçlarla karşılaştırma: [English README](README.md).
