# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603961
- **Họ và tên:** Hoàng Minh Quyền
- **SEED** (in ở ô đầu notebook): 4360

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính
 
| Cách | F1 (test) | F1 (SUB 40) | p95 ms/tin | USD / 1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.857 | 0.889 | 0.06 | 0 |
| 2. BoW + MLP | 0.942 | 0.889 | 0.39 | 0 |
| 3a. MiniLM + kNN | 0.897 | 0.974 | 35.66 | 0 |
| 3b. MiniLM + LogReg | **0.947** | 0.947 | 7.20 | 0 |
| 3c. MiniLM + MLP head | 0.940 | 0.974 | 7.33 | 0 |
| 4a. LLM nhỏ zero-shot (gpt-6-luna) | — | 0.923 | 2317 | 0.0068 |
| 4b. LLM lớn zero-shot (gpt-6.1-terra) | — | 0.947 | 1765 | 0.0678 |
| 4c. LLM nhỏ few-shot | — | 0.947 | 1887 | 0.0180 |
 
## 3. Dự đoán và thực tế
 
**`ml_lr` — lr = 50.** Mình đoán loss sẽ dao động răng cưa mãi, không hội tụ. Thực tế loss vọt lên ~2.5 ở vài epoch đầu rồi giảm rất nhanh xuống 0.006, thấp nhất trong 3 mức lr. Lệch vì loss của logistic regression là hàm lồi, gradient bị chặn (|p − y| ≤ 1), và dữ liệu bag-of-words gần như tách được tuyến tính nên loss không có "đáy" hữu hạn để nhảy qua lại. lr lớn chỉ làm trọng số phình to nhanh hơn theo đúng hướng, đổi lại mô hình quá tự tin (p gần 0 hoặc 1).
 
**`dl_relu` — bỏ ReLU.** Mình đoán F1 sẽ gần bằng logistic regression ở Phần 1 (0.857). Thực tế được 0.937. Về lý thuyết thì đúng là Linear + Linear vẫn là mô hình tuyến tính, nhưng khoảng cách đến từ cách tối ưu: Phần 1 là GD full-batch 300 bước (lr 0.5) chưa hội tụ, recall chỉ 0.764. Phần 2 dùng Adam, mini-batch, nên 30 epoch thực chất là hàng nghìn lần cập nhật. Bài học: muốn so 2 mô hình công bằng thì phải train cả hai tới hội tụ.
 
**`hard_eval` — 50 case khó.** Mình đoán LLM dễ bị prompt injection lừa và cần prompt phòng thủ. Thực tế cả 4a và 4d đều đúng 50/50, không bị lừa lần nào, nên prompt phòng thủ không cho thấy khác biệt. Ngoài ra mình nghĩ embedding nào cũng hơn BoW rõ, nhưng 3b và 3c chỉ đạt 62% so với 54% của BoW, còn kNN (68%) lại tốt nhất trong các model chạy cục bộ. Lệch vì gpt-6-luna đã đủ mạnh với kiểu injection trong bộ case này, còn đầu LogReg/MLP trên embedding vẫn học "spam" theo kiểu spam của dữ liệu train (trúng thưởng, nhạc chuông), nên không nhận ra lừa đảo kiểu mới.
 
## 4. Phân tích 50 case khó
 
BoW + LogReg tụt nhiều nhất: F1 từ 0.889 (SUB) xuống 0.378. Model này sai nặng ở nhóm thao túng/lừa đảo (đúng 12%), né từ khoá (17%) và spam giọng tự nhiên (20%). Ngược lại, nó đúng 100% ở nhóm "tưởng spam nhưng không phải" và "nói về spam", có lẽ vì các tin này có nhiều từ hội thoại (i, my, me, lol) kéo mạnh về ham.
 
- **#29 "C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd…"** (né từ khoá): tokenizer giữ nguyên "c0ngr4tulat10ns", "w0n", "c4rd" thành token lạ, không có trong vocab 2000 từ, nên trọng số bằng 0. Mất hết tín hiệu spam, bias âm kéo tin về ham. Embedding đúng 83% ở nhóm này vì tokenizer subword vẫn giữ được phần nào nghĩa.
- **#21 "Hi Mum, I dropped my phone in the toilet… send £200 to my friend's account"** (lừa đảo): toàn từ đời thường, lại có "my", "i", "mum", là những từ kéo về ham mạnh nhất. BoW chỉ đếm từ nên không hiểu ý đồ "đóng vai người thân để xin tiền". Chỉ LLM, vốn đọc hiểu được ý đồ, mới bắt được nhóm này một cách ổn định (100%).
## 5. Thử thách cá nhân
 
**Đề #5:** viết 4 tin (2 spam né từ khoá, 2 ham chứa "free"/"win"), so BoW với Embedding + LogReg.
 
| Tin | Thật | BoW | Emb (p spam) |
| --- | --- | --- | --- |
| 1. "Hello, your parcel is on hold due to unpaid shipping fee…" | spam | spam ✓ | ham ✗ (0.21) |
| 2. "Congratz, ur numb3r was pick3d 4 a c4sh r3ward, txt back Y…" | spam | spam ✓ | ham ✗ (0.16) |
| 3. "Are you free this Saturday? Wanna grab coffee after class" | ham | spam ✗ | ham ✓ (0.00) |
| 4. "We didn't win the game today but the team played really well…" | ham | ham ✓ | ham ✓ (0.00) |
 
BoW đúng 3/4, Embedding đúng 2/4. Kết quả ngược với dự đoán của mình (mình nghĩ embedding sẽ thắng ở 2 tin spam).
 
- **Tin 1:** BoW bắt được nhờ các từ như "your", "to", "please", "link", vốn hay đi với spam trong dữ liệu. Embedding hiểu nghĩa "thông báo giao hàng / thanh toán", nhưng dữ liệu train (spam SMS Anh đời cũ) gần như không có spam kiểu này. Vùng nghĩa đó toàn tin giao dịch bình thường nên LogReg gán ham.
- **Tin 2:** né được BoW với "numb3r", "c4sh", nhưng lại để lộ "txt", "ur", "4", "y", đều là token spam rất mạnh trong bộ dữ liệu. Với embedding, leetspeak bị cắt thành mảnh vô nghĩa, câu còn lại trông giống tin nhắn tán gẫu nên ra ham.
- **Tin 3:** "free" là một trong những từ có trọng số spam lớn nhất, BoW không hiểu "free" ở đây nghĩa là "rảnh" nên chặn nhầm. Embedding hiểu cả câu là lời rủ đi cà phê.
- **Tin 4:** cả hai đúng. "win" có trọng số yếu hơn "free", và các từ "team", "game", "practice" kéo về ham.
Rút ra: embedding hiểu nghĩa tốt hơn (tin 3), nhưng chỉ phân loại đúng khi dữ liệu train có loại spam đó. BoW né được khá dễ, nhưng spammer chỉ cần sót một token quen ("txt") là bị bắt.
 
## 6. Phương án cho sản phẩm thật
 
10 triệu SMS/ngày ≈ 116 tin/s trung bình, giờ cao điểm có thể gấp 5–10 lần. Yêu cầu p95 < 50 ms/tin.
 
**Chọn 3b (MiniLM + LogReg) làm bộ lọc chính.**
- F1 test cao nhất (0.947), p95 7.2 ms, nằm thoải mái dưới SLA, chi phí API bằng 0.
- Throughput đo trên Colab khoảng 1,334 tin/s cho cả lô, nên vài máy CPU là đủ chịu cao điểm.
- Trên 50 case khó được 62%, hơn BoW (54%).
**Vì sao không chọn các phương án khác:**
- **BoW + MLP** (F1 0.942, p95 0.39 ms) gần như hoà về F1 và rẻ hơn nhiều. Đây là phương án dự phòng tốt nếu thiếu tài nguyên, nhưng kém hơn trên case khó và dễ bị né từ khoá.
- **kNN** tốt nhất trên case khó (68%) nhưng p95 35.7 ms đã sát SLA. Với 10 triệu tin cần lưu, mỗi lần dự đoán phải so với cả kho vector, sẽ chậm và tốn bộ nhớ.
- **LLM** đúng 100% case khó, nhưng p95 khoảng 1.8–2.3 s, vượt SLA khoảng 40 lần. Chạy hết 10 triệu tin bằng gpt-6-luna tốn khoảng $67/ngày, bằng gpt-6.1-terra khoảng $680/ngày.
**Đánh đổi chấp nhận:** 3b vẫn yếu với lừa đảo kiểu mới và spam giọng tự nhiên (đúng 20–38% ở các nhóm này). Để bù, có thể chạy LLM **bất đồng bộ** (không chặn đường gửi tin) cho những tin 3b phân vân (p khoảng 0.3–0.7), dùng kết quả để gắn cờ và gán nhãn thêm dữ liệu, rồi train lại 3b định kỳ.
 
## 7. AI log và điều đã học
 
| Hỏi gì | Nhận gì | Mình sửa gì |
| --- | --- | --- |
| Viết các câu `predict` | AI đưa ra con số rất cụ thể | Viết lại theo giọng của mình, ước lượng mơ hồ hơn, không dùng số chính xác |
| Thử thách #5 | AI đề xuất 4 tin và dự đoán embedding thắng | Chạy thật thì kết quả ngược lại, tự phân tích lại ở mục 5 |
 
**Điều hiểu ra hôm nay:** điểm số trên tập test thường cho cảm giác rất an toàn, nhưng chỉ cần đổi sang dữ liệu "lạ" (50 case khó, hay 4 tin mình tự viết) là thứ hạng các mô hình thay đổi hẳn. Mô hình chỉ giỏi trong phạm vi dữ liệu nó từng thấy. Mình cũng hiểu rằng khi so hai mô hình, phải tách bạch chênh lệch do kiến trúc với chênh lệch do cách tối ưu (vụ bỏ ReLU vẫn đạt 0.937), và với tập nhỏ như SUB thì 1 tin đã đổi kết quả vài điểm, không nên xếp hạng theo số lẻ.
