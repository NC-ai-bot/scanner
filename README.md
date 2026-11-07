# 隨手掃描

手機拍照變成乾淨的 PDF。網址：https://nc-ai-bot.github.io/scanner/ （手機打開後「加入主畫面」就是 app，可離線使用）

- 影像處理用 OpenCV.js（`opencv.js`，在 `cvworker.js` 的 Web Worker 裡跑；管線在 `cvpipe.js`）
- 自動找紙張四角（Canny／Otsu／飽和度三路偵測）、透視拉正、光照校正去陰影、增強／灰階／黑白
- 去手寫：藍紅筆（色度）、鉛筆（連通元件無深色核心）、橡皮擦；AI 去手寫只在 Claude 版：https://claude.ai/artifact/U91zTDhgKzjACF264YRVuK
- 相機即時抓邊、對準自動連拍（PWA 版）；Claude 版用系統相機
- 頁面存在手機的 IndexedDB，照片不會上傳

改版時記得把 `sw.js` 的 `VER` 加一，手機才會拿到新版。

測試：`test/` 不在 repo 裡；用 Playwright + 假相機 y4m 跑過拍照、處理、匯出流程。
