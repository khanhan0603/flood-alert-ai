# Dua vietnam_flood_system len GitHub va bat CI

Thu muc `vietnam_flood_system` da co san GitHub Actions workflow tai `.github/workflows/ci.yml`.
Moi lan push len `main`/`master` hoac mo pull request, GitHub se:

1. Cai Python 3.11.
2. Cai dependency tu `requirements.txt`.
3. Kiem tra compile cac file trong `src/`.
4. Chay smoke test cho API va model trong `tests/`.

## Day project len repository

Chay cac lenh nay trong PowerShell:

```powershell
cd D:\ECMWFCode4Earth\vietnam_flood_system
git init
git branch -M main
git add .
git commit -m "Add Vietnam flood system with CI"
git remote add origin https://github.com/khanhan0603/flood-alert-ai.git
git push -u origin main
```

Neu remote da ton tai:

```powershell
git remote set-url origin https://github.com/khanhan0603/flood-alert-ai.git
git push -u origin main
```

## Luu y quan trong

`data/` dang duoc ignore vi thu muc nay rat lon. Neu can chia se data, hay dung GitHub Releases, Google Drive, DVC, hoac object storage thay vi commit truc tiep vao git.

Thu muc `models/` hien nho hon gioi han file thuong cua GitHub, nen co the commit de CI smoke test API. Neu model lon hon sau nay, nen chuyen model sang Git LFS hoac tai model tu release/artifact trong CI.

