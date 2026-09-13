"""
Ba agent cơ bản của vòng review (v1.4), đúng ba ô trong sơ đồ gốc: 1 Summary · 2 Filter · 3 Rank.

  summary.py   Summary agent: tư liệu đã truy hồi -> CulturalBrief (facts thị giác, khác gì với confusable, vẽ thế nào).
  describe.py  Filter agent: VLM MÔ TẢ ảnh -> so mô tả với must_have/must_not bằng văn bản -> giữ/bỏ có lý do.
  rank.py      Rank agent: xếp top-k ứng viên từ mô tả + brief; đối chiếu với xếp hạng theo metric ("debate" mức cơ bản).
  loop.py      Một vòng sửa: ứng viên đầu còn lỗi -> RevisionPlan -> sinh lại trên model tốt nhất -> lọc lại.

Vì sao tách "mô tả" khỏi "phán": v1.1 hỏi VLM 3B câu có/không trên ảnh thì nó trả "có" cho mọi câu. CULTIVate và Marmot
đều dùng VLM để trích mô tả rồi mới so với mô tả tham chiếu bằng văn bản; cách này ít thiên lệch hơn và kiểm được.
"""
