# Luồng giao tiếp K / O / R
Mỗi lời gọi ghi nguyên văn: system prompt (in một lần cho mỗi agent), user prompt, và JSON trả về. Dòng `may_kiem` / `luat_*` là bước kiểm bằng máy, không phải model.

---

## C001

### K · card_preservation  ·  10:33:17  ·  13.7s

<details><summary>system prompt của K</summary>

```
You are the Prompt Analyst. From the prompt ONLY, produce prompt_preservation: everything the prompt asks for, grouped. Keep meaning intact; the English wording in must_preserve_verbatim must be copied exactly. Needs no citation. JSON only.
```
</details>

<details><summary>user prompt</summary>

```
PROMPT VI: Ông đồ ngồi trên chiếu viết thư pháp bằng mực tàu lên giấy đỏ, phía sau là cổng Văn Miếu - Quốc Tử Giám.
PROMPT EN: An old Confucian scholar-calligrapher sits on a mat writing calligraphy in black ink on red paper, the gate of the Temple of Literature behind him.

Return prompt_preservation as JSON.
```
</details>

```json
{
 "prompt_preservation": {
  "main_entity": {
   "vi": "Ông đồ",
   "en": "An old Confucian scholar-calligrapher"
  },
  "subject_attributes": [
   "đồ ngồi trên chiếu",
   "viết thư pháp bằng mực tàu lên giấy đỏ",
   "sits on a mat",
   "writing calligraphy in black ink on red paper"
  ],
  "supporting_objects": [
   "mực tàu",
   "giấy đỏ",
   "mat",
   "black ink",
   "red paper"
  ],
  "background_and_scene": [
   "phía sau là cổng Văn Miếu - Quốc Tử Giám",
   "the gate of the Temple of Literature behind him"
  ],
  "actions_and_relations": [
   "ngồi trên chiếu",
   "viết thư pháp",
   "sits on a mat",
   "writing calligraphy"
  ],
  "colors": [
   "mực tàu",
   "giấy đỏ",
   "black ink",
   "red paper"
  ],
  "visual_effects": [],
  "time_weather_lighting": [],
  "counts": {},
  "text_in_image": [],
  "must_preserve_verbatim": [
   "Ông đồ",
   "đồ ngồi trên chiếu",
   "viết thư pháp bằng mực tàu lên giấy đỏ",
   "phía sau là cổng Văn Miếu - Quốc Tử Giám",
   "An old Confucian scholar-calligrapher",
   "sits on a mat",
   "writing calligraphy in black ink on red paper",
   "the gate of the Temple of Literature behind him"
  ]
 }
}
```

### K · card_cultural[ông đồ]  ·  10:33:37  ·  20.2s

<details><summary>system prompt của K</summary>

```
You are the Prompt & Cultural Analyst for a Vietnamese cultural text-to-image system.
You produce TWO cards as one JSON object.

CARD 1 — prompt_preservation: everything the prompt itself asks for, grouped. Take it ONLY from the prompt. Keep meaning intact. This card has the highest priority downstream. Needs no citation.

CARD 2 — cultural_evidence: ONLY culturally identifying features that a camera can capture, taken ONLY from the Wikipedia passages given to you. Rules:
- at most 3 identity_cues, at most 2 conditional_cues, at most 2 positive_disambiguators;
- every identity_cue and conditional_cue must carry quote_vi: a VERBATIM sentence or phrase copied exactly from the Vietnamese passage, and source_url from the passage header;
- if the passage does not clearly state a feature, do not invent it — list what you looked for under insufficient_evidence;
- history, etymology, symbolism, taste, smell, sound go to discarded_nonvisual_facts;
- a positive_disambiguator describes what the CORRECT object looks like where a look-alike would differ. Never describe the look-alike itself.
- Never turn a prompt-specific detail (e.g. 'white') into a universal cultural feature.
- Cues must describe the MAIN OBJECT's own appearance, or HOW it sits on the body (resting on one shoulder, worn over trousers, tied at the waist) when the source says so. Utensils, containers, serving temperature and preparation steps are not cues.
Answer in JSON only.
```
</details>

<details><summary>user prompt</summary>

```
PROMPT VI: Ông đồ ngồi trên chiếu viết thư pháp bằng mực tàu lên giấy đỏ, phía sau là cổng Văn Miếu - Quốc Tử Giám.
PROMPT EN: An old Confucian scholar-calligrapher sits on a mat writing calligraphy in black ink on red paper, the gate of the Temple of Literature behind him.

FOCUS ENTITY: ông đồ
Return cues ONLY for this entity (at most 2 identity_cues, 1 conditional_cue, 1 positive_disambiguator); set main_entity to it.

WIKIPEDIA PASSAGES (the only allowed source for cultural cues):
=== PASSAGE [vi] Ông đồ — https://vi.wikipedia.org/wiki/%C3%94ng_%C4%91%E1%BB%93 ===
Trong nền khoa cử Nho học, những người học sinh (anh khóa) đã thi qua 3 kỳ thi đỗ Tú Tài (trước 1828 gọi là Sinh đồ) - tên dân gian gọi là ông Đồ. Lúc đó, những sinh đồ tuy là đậu những kỳ thi cấp thấp hoặc thi không đỗ đạt, nhưng chưa đủ cao để được nhà nước quân chủ và phong kiến bổ làm quan, họ hoặc là phải học thêm để thi những kỳ thi cao hơn được tổ chức sau đó (như là thi Hội và thi Đình, xem thêm tại bài Khoa bảng Việt Nam), hay là tạm kiếm sống bằng những nghề dạy học (còn gọi là "thầy đồ"), viết thuê,... Trong thời chuyển đổi sang tân học, học chữ Quốc ngữ mà không còn học chữ Hán nữa, những ông đồ không có việc làm đã sinh sống bằng nghề viết chữ thuê, như trong bài thơ "Ông đồ" của Vũ Đình Liên. Mỗi năm hoa đào nở
Lại thấy ông đồ già
Bày mực tàu giấy đỏ
Bên phố đông người qua

Bao nhiêu người thuê viết
Tấm tắc ngợi thiên tài:
"Hoa tay thảo những nét
Như phượng múa, rồng bay"
....... Ông đồ, Vũ Đình Liên
Ngày nay, chữ "ông đồ" cũng được dùng để gọi những người có liên quan hay là có tiếp xúc với chữ Hán, với nền văn hóa Nho giáo, chẳng hạn những người viết chữ thư pháp hàng năm  vào dịp Tết hay là những người nghiên cứu Hán-Nôm. == Phố Ông Đồ ==
Ảnh hưởng của nền văn hóa Nho giáo về hình ảnh "Ông Đồ" cho đến nay vẫn còn được lưu giữ qua hình ảnh các "Phố Ông Đồ" tại Hà Nội và Thành phố Hồ Chí Minh, nhằm tôn vinh và gìn giữ một giá trị văn hóa truyền thống của người Việt. Nhận xét về các "Phố Ông Đồ" này, Nhà báo, Thạc sĩ văn hóa Nguyễn Thành Luân cho rằng, cũng giống như không gian hồ Văn, Văn Miếu - Quốc Tử Giám (Hà Nội) được biết đến như "Làng sĩ tử" ở phía Bắc thì ở Phố Ông Đồ ở Thành phố Hồ Chí Minh (tọa lạc tại góc đường Nguyễn Thị Minh Khai và Phạm Ngọc Thạch, quận 1) cũng có những nét riêng có rất độc đáo. Về lịch sử của Phố Ông Đồ tại Thành phố Hồ Chí Minh gắn liền cùng với tục xin chữ ở vùng đất Nam bộ bắt nguồn từ thế kỷ thứ 17, tức có từ thời chúa Nguyễn vào khai phá vùng đất phương Nam, cùng với các đợt người Việt (người Kinh từ miền Bắc, miền Trung) di cư vào theo. Ông Đồ được nhắc đến trong một bài thơ cùng tên của Vũ Đình Liên với một cảm xúc đầy tiếc nuối cho một di sản văn hóa Việt Nam như ông Đồ bị lụi tàn.

Return cultural_evidence as JSON. quote_vi must be copied verbatim.
```
</details>

```json
{
 "cultural_evidence": {
  "identity_cues": [
   {
    "cue_vi": "mực tàu",
    "cue_en": "black inkstick",
    "visual_check_vi": "một cây mực tàu",
    "quote_vi": "Bày mực tàu giấy đỏ",
    "source_url": "https://vi.wikipedia.org/wiki/%C3%94ng_%C4%91%E1%BB%93",
    "entity": "ông đồ"
   },
   {
    "cue_vi": "giấy đỏ",
    "cue_en": "red paper",
    "visual_check_vi": "một tờ giấy đỏ",
    "quote_vi": "Bày mực tàu giấy đỏ",
    "source_url": "https://vi.wikipedia.org/wiki/%C3%94ng_%C4%91%E1%BB%93",
    "entity": "ông đồ"
   }
  ],
  "conditional_cues": [
   {
    "cue_vi": "đang viết thư pháp",
    "cue_en": "writing calligraphy",
    "visual_check_vi": "người đang viết thư pháp",
    "quote_vi": "những ông đồ không có việc làm đã sinh sống bằng nghề viết chữ thuê",
    "source_url": "https://vi.wikipedia.org/wiki/%C3%94ng_%C4%91%E1%BB%93",
    "entity": "ông đồ"
   }
  ],
  "positive_disambiguators": [
   {
    "target_appearance_vi": "đang viết thư pháp",
    "target_appearance_en": "writing calligraphy",
    "source_url": "https://vi.wikipedia.org/wiki/%C3%94ng_%C4%91%E1%BB%93",
    "entity": "ông đồ"
   }
  ],
  "discarded_nonvisual_facts": [
   "nền khoa cử Nho học",
   "thời chuyển đổi sang tân học",
   "nghề viết chữ thuê",
   "những người có liên quan hay là có tiếp xúc với chữ Hán, với nền văn hóa Nho giáo",
   "tục xin chữ ở vùng đất Nam bộ bắt nguồn từ thế kỷ thứ 17",
   "cảm xúc đầy tiếc nuối cho một di sản văn hóa Việt Nam như ông Đồ bị lụi tàn"
  ],
  "insufficient_evidence": [
   "trang phục của ông đồ",
   "màu sắc của trang phục",
   "cách ngồi của ông đồ"
  ]
 }
}
```

### K · card_cultural[thư pháp mực tàu]  ·  10:33:58  ·  21.2s

<details><summary>user prompt</summary>

```
PROMPT VI: Ông đồ ngồi trên chiếu viết thư pháp bằng mực tàu lên giấy đỏ, phía sau là cổng Văn Miếu - Quốc Tử Giám.
PROMPT EN: An old Confucian scholar-calligrapher sits on a mat writing calligraphy in black ink on red paper, the gate of the Temple of Literature behind him.

FOCUS ENTITY: thư pháp mực tàu
Return cues ONLY for this entity (at most 2 identity_cues, 1 conditional_cue, 1 positive_disambiguator); set main_entity to it.

WIKIPEDIA PASSAGES (the only allowed source for cultural cues):
=== PASSAGE [vi] Thư pháp — https://vi.wikipedia.org/wiki/Th%C6%B0_ph%C3%A1p ===
Thư pháp là một từ có nghĩa "phương pháp viết chữ", thường được hiểu là các phương pháp, cách thức viết chữ đẹp mắt. Thư pháp gia (Hán văn: 書法家, Hi văn: Καλλιγράφος) được hiểu là những người thạo thư pháp và có khả năng đẩy văn tự lên tầm nghệ thuật. == Lịch sử ==

Thư pháp xuất hiện hầu như đồng thời với sự hình thành văn tự vì nhu cầu trình bày trang nhã và minh bạch văn bản. Công cụ chính tạo nên thư pháp là bút và chất liệu lưu trữ văn tự, mà sau được hiểu gồm mực và giấy. Tại phương Đông, người viết chữ cũng thường sử dụng nghiên mực để chứa mực, vì vậy bốn món đồ bao gồm: Bút, Nghiên, Giấy, Mực được xem là "Văn Phòng Tứ Bảo" (4 món đồ quan trọng bậc nhất của người viết chữ). Thư pháp gia thường không nhất thiết phải là bậc trí giả nhưng ít nhiều được coi trọng nhờ vốn học vấn đủ để biên chép. Trong thế giới Tây phương, thư pháp gia thường là hiền sĩ hoặc tăng lữ vì đặc thù thực dụng của thể chữ kí ngang mặt giấy. Ngược lại, ở thế giới Á Đông, thư pháp gia có sự phân cấp rất rõ : Thư pháp gia đã ít nhất một lần đậu khoa cử hoặc được kính nể cả tài năng và đức hạnh thì mặc nhiên liệt hạng tinh anh, những người này coi chữ của mình là vốn quý và chỉ đem tặng những ai thân thiết hay khả kính ; còn lại, thư pháp gia vốn do thi hỏng thì chẳng những không đủ tư cách dạy học mà còn bị cả sĩ lâm coi khinh, vì thế, đối với họ thì bán chữ thuần túy là cách mưu sinh. Những thư pháp gia có địa vị thấp thường bị lỡm là "vị sư bán tự" (lái "bán tự vi sư"), thoảng hoặc kiêm thêm nghề bói toán hoặc bốc thuốc làm kế sinh nhai.. Tại Việt Nam hậu kì hiện đại, do tình trạng kì thị cũng như đứt đoạn văn hóa Hán tự, nhiều người tự xưng thư pháp gia đi bán chữ theo lối cổ, nhưng thường phạm những điều kị của giới thư họa truyền thống : Không phân biệt được các thể Hán tự, không biết cách cầm bút, pha mực, thậm chí lạm dụng kí tự Latin vào thư pháp Hán (do bản thân họ không học Hán tự nghiêm túc)... Lớp người này bị giới thư họa truyền thống lên án và phê phán rất gay gắt. == Phong hóa ==

Ở Tây Âu trung đại, trang sức thủ bản (manuscriptum illuminatum) là trường phái mĩ thuật kết hợp giữa thư pháp và hội họa, đòi công phu cùng chất liệu tương đối đắt. Thư pháp gia thường là những linh mục có tài nghệ đặc biệt được chủng viện giao phó soạn nội dung liên quan đến thánh tích, các điều hay lạ trong dân gian cùng thời. Thư pháp Á Đông được cho là hình thành từ thời Đông Châu Liệt Quốc với các thể kim văn, triện văn bắt nguồn từ giáp cốt văn.

Return cultural_evidence as JSON. quote_vi must be copied verbatim.
```
</details>

```json
{
 "cultural_evidence": {
  "identity_cues": [
   {
    "cue_vi": "mực tàu",
    "cue_en": "ink stick",
    "visual_check_vi": "một cây mực tàu",
    "quote_vi": "Công cụ chính tạo nên thư pháp là bút và chất liệu lưu trữ văn tự, mà sau được hiểu gồm mực và giấy.",
    "source_url": "https://vi.wikipedia.org/wiki/Th%C6%B0_ph%C3%A1p",
    "entity": "thư pháp mực tàu"
   },
   {
    "cue_vi": "giấy đỏ",
    "cue_en": "red paper",
    "visual_check_vi": "một tờ giấy đỏ",
    "quote_vi": "Công cụ chính tạo nên thư pháp là bút và chất liệu lưu trữ văn tự, mà sau được hiểu gồm mực và giấy.",
    "source_url": "https://vi.wikipedia.org/wiki/Th%C6%B0_ph%C3%A1p",
    "entity": "thư pháp mực tàu"
   }
  ],
  "conditional_cues": [
   {
    "cue_vi": "bút",
    "cue_en": "brush",
    "visual_check_vi": "một cây bút",
    "quote_vi": "Công cụ chính tạo nên thư pháp là bút và chất liệu lưu trữ văn tự, mà sau được hiểu gồm mực và giấy.",
    "source_url": "https://vi.wikipedia.org/wiki/Th%C6%B0_ph%C3%A1p",
    "entity": "thư pháp mực tàu"
   }
  ],
  "positive_disambiguators": [
   {
    "target_appearance_vi": "một cây bút có lông mềm",
    "target_appearance_en": "a brush with soft bristles",
    "source_url": "https://vi.wikipedia.org/wiki/Th%C6%B0_ph%C3%A1p",
    "entity": "thư pháp mực tàu"
   }
  ],
  "discarded_nonvisual_facts": [
   "Thư pháp xuất hiện hầu như đồng thời với sự hình thành văn tự vì nhu cầu trình bày trang nhã và minh bạch văn bản.",
   "Thư pháp gia thường không nhất thiết phải là bậc trí giả nhưng ít nhiều được coi trọng nhờ vốn học vấn đủ để biên chép.",
   "Thư pháp Á Đông được cho là hình thành từ thời Đông Châu Liệt Quốc với các thể kim văn, triện văn bắt nguồn từ giáp cốt văn."
  ],
  "insufficient_evidence": [
   "the color of the ink stick",
   "the color of the brush",
   "the size of the paper"
  ]
 }
}
```

### K · card_cultural[Văn Miếu - Quốc Tử Giám]  ·  10:34:25  ·  27.3s

<details><summary>user prompt</summary>

```
PROMPT VI: Ông đồ ngồi trên chiếu viết thư pháp bằng mực tàu lên giấy đỏ, phía sau là cổng Văn Miếu - Quốc Tử Giám.
PROMPT EN: An old Confucian scholar-calligrapher sits on a mat writing calligraphy in black ink on red paper, the gate of the Temple of Literature behind him.

FOCUS ENTITY: Văn Miếu - Quốc Tử Giám
Return cues ONLY for this entity (at most 2 identity_cues, 1 conditional_cue, 1 positive_disambiguator); set main_entity to it.

WIKIPEDIA PASSAGES (the only allowed source for cultural cues):
=== PASSAGE [vi] Văn Miếu – Quốc Tử Giám — https://vi.wikipedia.org/wiki/V%C4%83n_Mi%E1%BA%BFu_%E2%80%93_Qu%E1%BB%91c_T%E1%BB%AD_Gi%C3%A1m ===
Văn Miếu – Quốc Tử Giám là quần thể di tích đa dạng, phong phú hàng đầu của thành phố Hà Nội, nằm ở phía Nam kinh thành Thăng Long. Quần thể kiến trúc Văn Miếu – Quốc Tử Giám bao gồm: hồ Văn, khu Văn Miếu – Quốc Tử Giám và vườn Giám, mà kiến trúc chủ thể là Văn miếu (chữ Hán: 文廟) - nơi thờ Khổng Tử, và Quốc tử giám (chữ Hán: 國子監) - trường đại học đầu tiên của Việt Nam. Khu Văn Miếu – Quốc Tử Giám có tường gạch vồ bao quanh, phía trong chia thành 5 lớp không gian với các kiến trúc khác nhau. Từ ngoài vào trong có các cổng lần lượt là: cổng Văn Miếu, Đại Trung, Khuê Văn các, Đại Thành và cổng Thái Học. Ngày nay, Văn Miếu – Quốc Tử Giám là nơi tham quan của du khách trong và ngoài nước đồng thời cũng là nơi khen tặng cho học sinh xuất sắc và còn là nơi tổ chức hội thơ hàng năm vào ngày rằm tháng giêng. == Lịch sử ==
Văn miếu được xây dựng từ năm 1070, tức năm Thần Vũ thứ hai đời Lý Thánh Tông. Khoa học xã hội, Hà Nội, tập 1, tr.234 chép: "Mùa thu tháng 10, làm Văn Miếu, đắp tượng, Khổng Tử, Chu Công và Tứ phối, vẽ tượng Thất thập nhị hiền, bốn mùa cúng tế. Hoàng thái tử đến đấy học."  Như vậy Văn miếu ngoài chức năng thờ các bậc Tiên thánh, Tiên sư của đạo Nho, còn là một trường học Hoàng gia mà học trò đầu tiên là Thái tử Lý Càn Đức, con trai vua Lý Thánh Tông với Nguyên phi Ỷ Lan, lúc đó mới 5 tuổi, đến năm 1072 lên ngôi trở thành vua Lý Nhân Tông. Năm 1076, Lý Nhân Tông cho lập trường Quốc Tử giám ở bên cạnh Văn miếu, có thể coi đây là trường đại học đầu tiên ở Việt Nam. Ban đầu, trường chỉ dành riêng cho con vua và con các bậc đại quyền quý (nên gọi tên là Quốc tử). (Việt sử thông giám cương mục. Văn sử địa. lập nhà Quốc tử giám; tuyển trong các văn thần lấy những người có văn học, bổ vào đó". Năm 1156, Lý Anh Tông cho sửa lại Văn miếu và chỉ thờ Khổng Tử. Năm 1253, vua Trần Thái Tông đổi Quốc Tử giám thành Quốc Học viện cho mở rộng và thu nhận cả con cái các nhà thường dân có sức học xuất sắc. Chức năng trường Quốc học ngày càng nổi bật hơn chức năng của một nơi tế lễ "Quý Sửu năm thứ ba (1253)... Tháng 6 lập Quốc Học viện, tô tượng Khổng Tử, Chu công và Á Thánh, vẽ tượng 72 người hiền để thờ... Tháng 9 xuống chiếu cho các nho sĩ trong nước đến Quốc học viện giảng học tứ thư, lục kinh" (ĐVSKTT). Lấy Phạm Ứng Thần giữ chức Thượng thư kiêm chức Đề điệu Quốc Tử viện để trông nom công việc học tập tại Quốc Tử giám". Đời Trần Minh Tông, Chu Văn An được cử làm quan Quốc Tử giám Tư nghiệp (hiệu trưởng) và thầy dạy trực tiếp của các hoàng tử.

Return cultural_evidence as JSON. quote_vi must be copied verbatim.
```
</details>

```json
{
 "cultural_evidence": {
  "identity_cues": [
   {
    "cue_vi": "cổng Văn Miếu",
    "cue_en": "gate of the Temple of Literature",
    "visual_check_vi": "cổng có tên là cổng Văn Miếu",
    "quote_vi": "cổng Văn Miếu",
    "source_url": "https://vi.wikipedia.org/wiki/V%C4%83n_Mi%E1%BA%BFu_%E2%80%93_Qu%E1%BB%91c_T%E1%BB%AD_Gi%C3%A1m",
    "entity": "Văn Miếu - Quốc Tử Giám"
   },
   {
    "cue_vi": "cổng Thái Học",
    "cue_en": "Thai Hoc Gate",
    "visual_check_vi": "cổng có tên là cổng Thái Học",
    "quote_vi": "cổng Thái Học",
    "source_url": "https://vi.wikipedia.org/wiki/V%C4%83n_Mi%E1%BA%BFu_%E2%80%93_Qu%E1%BB%91c_T%E1%BB%AD_Gi%C3%A1m",
    "entity": "Văn Miếu - Quốc Tử Giám"
   }
  ],
  "conditional_cues": [
   {
    "cue_vi": "tường gạch vồ bao quanh",
    "cue_en": "surrounded by a brick wall",
    "visual_check_vi": "có tường gạch vồ bao quanh",
    "quote_vi": "Khu Văn Miếu – Quốc Tử Giám có tường gạch vồ bao quanh",
    "source_url": "https://vi.wikipedia.org/wiki/V%C4%83n_Mi%E1%BA%BFu_%E2%80%93_Qu%E1%BB%91c_T%E1%BB%AD_Gi%C3%A1m",
    "entity": "Văn Miếu - Quốc Tử Giám"
   }
  ],
  "positive_disambiguators": [
   {
    "target_appearance_vi": "có nhiều cổng",
    "target_appearance_en": "has many gates",
    "source_url": "https://vi.wikipedia.org/wiki/V%C4%83n_Mi%E1%BA%BFu_%E2%80%93_Qu%E1%BB%91c_T%E1%BB%AD_Gi%C3%A1m",
    "entity": "Văn Miếu - Quốc Tử Giám"
   }
  ],
  "discarded_nonvisual_facts": [
   "Văn Miếu – Quốc Tử Giám là quần thể di tích đa dạng, phong phú hàng đầu của thành phố Hà Nội, nằm ở phía Nam kinh thành Thăng Long.",
   "Văn miếu được xây dựng từ năm 1070, tức năm Thần Vũ thứ hai đời Lý Thánh Tông.",
   "Văn miếu ngoài chức năng thờ các bậc Tiên thánh, Tiên sư của đạo Nho, còn là một trường học Hoàng gia.",
   "Năm 1076, Lý Nhân Tông cho lập trường Quốc Tử giám ở bên cạnh Văn miếu, có thể coi đây là trường đại học đầu tiên ở Việt Nam.",
   "Ngày nay, Văn Miếu – Quốc Tử Giám là nơi tham quan của du khách trong và ngoài nước đồng thời cũng là nơi khen tặng cho học sinh xuất sắc và còn là nơi tổ chức hội thơ hàng năm vào ngày rằm tháng giêng."
  ],
  "insufficient_evidence": [
   "the appearance of the gates"
  ]
 }
}
```

### K · may_kiem  ·  10:34:25

*(kiểm bằng máy: quote_vi phải nguyên văn trong bài vi)*

```json
{
 "giu_identity": 5,
 "giu_conditional": 3,
 "loai": [
  "identity_cues: 'giấy đỏ' — quote có thật nhưng không nói về cue"
 ]
}
```

### O · I0  ·  10:34:40  ·  14.3s

<details><summary>system prompt của O</summary>

```
You are a Blind Visual Observer. You receive ONE photograph and nothing else. You do not know what it was supposed to show.
Report EXHAUSTIVELY what is visible, in this order: 1 main subjects; 2 clothing and appearance; 3 foreground objects; 4 supporting objects; 5 background; 6 actions and spatial relations; 7 colours; 8 counts; 9 small effects such as smoke, steam, light, motion, reflections; 10 uncertain regions.
RULES: describe only what a camera captured — shapes, materials, how parts join, counts, colours. Never say whether anything is correct, authentic or traditional. Never name a country, culture or ethnicity. Never guess at anything outside the frame; list cut-off parts under uncertain. Short noun phrases, 3-12 words each. JSON only.
```
</details>

ảnh vào: `C001_sdxl_base_c0.png`

<details><summary>user prompt</summary>

```
Describe this photograph. Return JSON.
```
</details>

```json
{
 "main_subjects": [
  "person sitting on small stool",
  "large tree trunk",
  "large white paper on ground"
 ],
 "clothing_and_appearance": [
  "person wearing dark blue robe",
  "person wearing glasses",
  "person holding red brush"
 ],
 "foreground_objects": [
  "white paper with black writing",
  "small red stool",
  "red brush on ground",
  "stack of papers on ground"
 ],
 "supporting_objects": [
  "large tree trunk with stone base",
  "small brown jar on ground",
  "red building with tiled roof"
 ],
 "background": [
  "red building with tiled roof",
  "large tree with green leaves",
  "gray sky with mist"
 ],
 "actions": [
  "person writing on paper",
  "person holding brush"
 ],
 "spatial_relations": [
  "person sitting in front of tree",
  "paper on ground in front of person",
  "jar on ground near paper",
  "stool under person",
  "brush in person's hand"
 ],
 "colors": [
  "dark blue robe",
  "red brush and stool",
  "white paper",
  "black writing on paper",
  "green leaves",
  "red building",
  "gray sky"
 ],
 "counts": {
  "people": 1,
  "trees": 1,
  "brushes": 2,
  "jars": 1,
  "stools": 1
 },
 "visible_effects": [
  "mist in background",
  "shadows on ground",
  "light reflecting off paper"
 ],
 "uncertain": [
  "top of tree",
  "far right side of building",
  "bottom of person's robe",
  "far left side of paper"
 ]
}
```

### R · gap  ·  10:34:57  ·  17.6s

<details><summary>system prompt của R</summary>

```
You are the Gap Analyzer & Prompt Refiner. You compare a blind visual report of a generated image against (a) the Prompt Preservation Card and (b) the Cultural Evidence Card.
Classify: missing_prompt_explicit (asked by the prompt, not in the report); missing_cultural_identity (identity cue not evidenced in the report); contradictions (report shows something incompatible); already_satisfied; uncertain_no_repair (the report is unsure — do NOT repair these).
Priority for repair: 1 details stated in the prompt; 2 supporting objects, background, relations; 3 identity cues; 4 conditional cues. At most 3 repair_actions.
Each repair_action is ONE positive English instruction (max 18 words) describing exactly what should be visibly present, including WHERE on the body or scene it sits when that matters (e.g. 'a bamboo pole resting across one shoulder with a basket hanging from each end'). Never use negation (no/not/without/avoid). Never name the wrong object. Never change framing, camera angle, lighting or composition. Do not restate things already satisfied.
drop_phrases: copy VERBATIM any phrase from the EXPANSION TEXT (never from the original prompt) that contradicts the Cultural Evidence Card or pushes toward a look-alike object — e.g. a cut or garment term that belongs to another culture. Empty list if none. JSON only.
```
</details>

<details><summary>user prompt</summary>

```
ORIGINAL PROMPT: An old Confucian scholar-calligrapher sits on a mat writing calligraphy in black ink on red paper, the gate of the Temple of Literature behind him.

EXPANSION TEXT (added by Culture-TRIP, may be trimmed):
In the serene courtyard of the Văn Miếu - Quốc Tử Giám in Hanoi, Vietnam, an elderly Confucian scholar, known as an 'ông đồ', sits on a bamboo mat. Dressed in a long, flowing robe adorned with intricate embroidery unique to Vietnamese culture, he meticulously writes calligraphy on mulberry paper using a traditional brush. Surrounding him, blooming flowers and lush greenery create a tranquil backdrop. On his table, ancient calligraphy tools such as an ink stone, a water dropper, and a stack of brush pens are neatly arranged. The gentle rustle of leaves, the subtle scent of ink, and the rhythmic sound of the brush sweeping across the paper create a meditative atmosphere, inviting observers to appreciate the deep cultural and historical significance of the scholar's work. Just before the Tết Vietnamese New Year celebration, the courtyard transforms into a gathering place for calligraphists who write wishes in Chữ Hán, the classical script of Vietnam. These calligraphic works are given away as gifts or used as home decorations for special occasions, reflecting the importance of calligraphy in Vietnamese culture and its role in conveying good wishes and auspiciousness.

PROMPT PRESERVATION CARD:
{"main_entity": {"vi": "Ông đồ", "en": "An old Confucian scholar-calligrapher"}, "subject_attributes": ["đồ ngồi trên chiếu", "viết thư pháp bằng mực tàu lên giấy đỏ", "sits on a mat", "writing calligraphy in black ink on red paper"], "supporting_objects": ["mực tàu", "giấy đỏ", "mat", "black ink", "red paper"], "background_and_scene": ["phía sau là cổng Văn Miếu - Quốc Tử Giám", "the gate of the Temple of Literature behind him"], "actions_and_relations": ["ngồi trên chiếu", "viết thư pháp", "sits on a mat", "writing calligraphy"], "colors": ["mực tàu", "giấy đỏ", "black ink", "red paper"], "visual_effects": [], "time_weather_lighting": [], "counts": {}, "text_in_image": [], "must_preserve_verbatim": ["Ông đồ", "đồ ngồi trên chiếu", "viết thư pháp bằng mực tàu lên giấy đỏ", "phía sau là cổng Văn Miếu - Quốc Tử Giám", "An old Confucian scholar-calligrapher", "sits on a mat", "writing calligraphy in black ink on red paper", "the gate of the Temple of Literature behind him"]}

CULTURAL EVIDENCE CARD:
{"identity_cues": [{"cue_vi": "mực tàu", "cue_en": "black inkstick", "visual_check_vi": "một cây mực tàu", "quote_vi": "Bày mực tàu giấy đỏ", "source_url": "https://vi.wikipedia.org/wiki/%C3%94ng_%C4%91%E1%BB%93", "entity": "ông đồ"}, {"cue_vi": "giấy đỏ", "cue_en": "red paper", "visual_check_vi": "một tờ giấy đỏ", "quote_vi": "Bày mực tàu giấy đỏ", "source_url": "https://vi.wikipedia.org/wiki/%C3%94ng_%C4%91%E1%BB%93", "entity": "ông đồ"}, {"cue_vi": "mực tàu", "cue_en": "ink stick", "visual_check_vi": "một cây mực tàu", "quote_vi": "Công cụ chính tạo nên thư pháp là bút và chất liệu lưu trữ văn tự, mà sau được hiểu gồm mực và giấy.", "source_url": "https://vi.wikipedia.org/wiki/Th%C6%B0_ph%C3%A1p", "entity": "thư pháp mực tàu"}, {"cue_vi": "cổng Văn Miếu", "cue_en": "gate of the Temple of Literature", "visual_check_vi": "cổng có tên là cổng Văn Miếu", "quote_vi": "cổng Văn Miếu", "source_url": "https://vi.wikipedia.org/wiki/V%C4%83n_Mi%E1%BA%BFu_%E2%80%93_Qu%E1%BB%91c_T%E1%BB%AD_Gi%C3%A1m", "entity": "Văn Miếu - Quốc Tử Giám"}, {"cue_vi"
… [cắt, tổng 6952 ký tự]
```
</details>

```json
{
 "missing_prompt_explicit": [
  "sits on a mat",
  "writing calligraphy in black ink on red paper",
  "the gate of the Temple of Literature behind him"
 ],
 "missing_cultural_identity": [
  "black inkstick",
  "red paper",
  "cổng Văn Miếu",
  "cổng Thái Học"
 ],
 "contradictions": [
  "person sitting on small stool",
  "white paper with black writing",
  "white paper on ground",
  "paper on ground in front of person",
  "stack of papers on ground",
  "person wearing dark blue robe",
  "red brush",
  "red brush on ground",
  "brush in person's hand",
  "large tree trunk",
  "large tree with green leaves",
  "person sitting in front of tree",
  "small brown jar on ground",
  "jar on ground near paper",
  "red building with tiled roof"
 ],
 "already_satisfied": [
  "person writing on paper",
  "person holding brush"
 ],
 "uncertain_no_repair": [
  "top of tree",
  "far right side of building",
  "bottom of person's robe",
  "far left side of paper"
 ],
 "repair_actions": [
  "a bamboo mat under the person",
  "a black inkstick on the ground near the person",
  "the gate of the Temple of Literature behind the person"
 ],
 "drop_phrases": [
  "long, flowing robe",
  "mulberry paper",
  "ancient calligraphy tools such as an ink stone, a water dropper, and a stack of brush pens"
 ]
}
```

### R · may_kiem  ·  10:34:57

*(kiểm bằng máy: bỏ action có phủ định / khung hình; trần 3; drop_phrases phải nguyên văn trong phần mở rộng)*

```json
{
 "giu": [
  "a bamboo mat under the person",
  "a black inkstick on the ground near the person",
  "the gate of the Temple of Literature behind the person"
 ],
 "bo": [],
 "drop_phrases": [
  "long, flowing robe",
  "mulberry paper",
  "ancient calligraphy tools such as an ink stone, a water dropper, and a stack of brush pens"
 ]
}
```

### O · pre_I0  ·  10:34:58  ·  1.3s

<details><summary>system prompt của O</summary>

```
You rate how well ONE photograph matches ONE written statement. score 0-10: 10 exactly as described, 5 the thing is there but clearly differs, 0 nothing of the kind. If the part lies outside the picture set in_frame false and give no score. Never name a country or culture. evidence: what you see, max 15 words. JSON only.
```
</details>

ảnh vào: `C001_sdxl_base_c0.png`

<details><summary>user prompt</summary>

```
Statement: "a bamboo mat under the person"
How fully does this photograph match that statement?
Return JSON: {"in_frame": true|false, "score": 0-10, "evidence": ".."}
```
</details>

```json
{
 "in_frame": true,
 "score": 0.0,
 "evidence": "no bamboo mat visible"
}
```

### O · pre_I0  ·  10:35:00  ·  1.2s

ảnh vào: `C001_sdxl_base_c0.png`

<details><summary>user prompt</summary>

```
Statement: "a black inkstick on the ground near the person"
How fully does this photograph match that statement?
Return JSON: {"in_frame": true|false, "score": 0-10, "evidence": ".."}
```
</details>

```json
{
 "in_frame": true,
 "score": 5.0,
 "evidence": "black object near person"
}
```

### O · pre_I0  ·  10:35:01  ·  1.3s

ảnh vào: `C001_sdxl_base_c0.png`

<details><summary>user prompt</summary>

```
Statement: "the gate of the Temple of Literature behind the person"
How fully does this photograph match that statement?
Return JSON: {"in_frame": true|false, "score": 0-10, "evidence": ".."}
```
</details>

```json
{
 "in_frame": true,
 "score": 5.0,
 "evidence": "building behind person, no gate"
}
```

### R · may_kiem_I0  ·  10:35:01

*(chấm action trên I0; bỏ mục đã >= 8; bù identity cue của K)*

```json
{
 "giu": [
  [
   "a bamboo mat under the person",
   0.0
  ],
  [
   "a black inkstick on the ground near the person",
   5.0
  ],
  [
   "the gate of the Temple of Literature behind the person",
   5.0
  ]
 ],
 "bo_da_dat": []
}
```

### P1 · prompt  ·  10:35:01

*(P1 = P_ct nguyên văn + preserve clause + action; không negative)*

```json
{
 "P1": "An old Confucian scholar-calligrapher sits on a mat writing calligraphy in black ink on red paper, the gate of the Temple of Literature behind him. In the serene courtyard of the Văn Miếu - Quốc Tử Giám in Hanoi, Vietnam, an elderly Confucian scholar, known as an 'ông đồ', sits on a bamboo mat. Dressed in a adorned with intricate embroidery unique to Vietnamese culture, he meticulously writes calligraphy on using a traditional brush. Surrounding him, blooming flowers and lush greenery create a tranquil backdrop. On his table, are neatly arranged. The gentle rustle of leaves, the subtle scent of ink, and the rhythmic sound of the brush sweeping across the paper create a meditative atmosphere, inviting observers to appreciate the deep cultural and historical significance of the scholar's work. Just before the Tết Vietnamese New Year celebration, the courtyard transforms into a gathering place for calligraphists who write wishes in Chữ Hán, the classical script of Vietnam. These calligraphic works are given away as gifts or used as home decorations for special occasions, reflecting the importance of calligraphy in Vietnamese culture and its role in conveying good wishes and auspiciousness.\n\nPreserve the current subject, composition, setting, colors, and all correctly rendered details. Keep clearly visible: mat; black ink; red paper; the gate of the Temple of Literature behind him. Make these additions clearly visible:\n1. a bamboo mat under the person.\n2. a black inkstick on the ground near the person.\n3. the gate of the Temple of Literature behind the person."
}
```
