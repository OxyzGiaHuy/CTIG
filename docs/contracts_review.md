# Contract — đúng thứ Critic đọc
Sửa `data/contracts.json` là đổi hẳn thứ hệ thống đi tìm; không cần đụng vào mã.
Ba chỗ đáng soi: **mục nào không có bộ phận tương ứng** thì Observer sẽ không nhắc tới và Critic không bao giờ bắt được lỗi ở đó; **mục CHƯA CÓ NGUỒN**; và **mục không phải lúc nào cũng xuất hiện** — mục kiểu đó làm Critic bắt lỗi oan rồi hệ thống sửa hỏng ảnh vốn đúng.

---

## S001 · áo dài — *Vietnamese ao dai*
> prompt gốc: A young woman in a white ao dai standing at a school gate.

**Observer được chỉ soi:** `slits, panels, trousers`

**Critic đọc nguyên văn:**

```
VISUAL CONTRACT for Vietnamese ao dai
REQUIRED (each must be visible):
  - id=side_slits_at_waist: side slits from the waist making front and back panels
  - id=worn_over_trousers: worn over separate long wide-legged trousers
  - id=ankle_length_panels: loose panels hanging down to near the ankles
CONFUSABLE OBJECTS (must NOT be what is shown):
  - id=qipao_cheongsam: a one-piece fitted dress with one thigh slit, no trousers
  - id=kimono: a wrapped robe with crossed collar and wide obi sash
  - id=hanbok: a short jacket over a wide high-waisted flared skirt
```

| mục | mô tả | có trong danh sách soi? | nguồn |
|---|---|---|---|
| `side_slits_at_waist` | side slits from the waist making front and back panels | có | https://vi.wikipedia.org/wiki/%C3%81o_d%C3%A0i |
| `worn_over_trousers` | worn over separate long wide-legged trousers | có | https://vi.wikipedia.org/wiki/%C3%81o_d%C3%A0i |
| `ankle_length_panels` | loose panels hanging down to near the ankles | có | https://vi.wikipedia.org/wiki/%C3%81o_d%C3%A0i |
| ~~`qipao_cheongsam`~~ dễ nhầm | a one-piece fitted dress with one thigh slit, no trousers | — | Chinese |
| ~~`kimono`~~ dễ nhầm | a wrapped robe with crossed collar and wide obi sash | — | Japanese |
| ~~`hanbok`~~ dễ nhầm | a short jacket over a wide high-waisted flared skirt | — | Korean |

---

## S002 · gánh hàng rong — *Vietnamese shoulder-pole street vendor*
> prompt gốc: A woman carrying a shoulder pole of street goods along a street in the morning.

**Observer được chỉ soi:** `pole, shoulder, baskets, trays`

**Critic đọc nguyên văn:**

```
VISUAL CONTRACT for Vietnamese shoulder-pole street vendor
REQUIRED (each must be visible):
  - id=pole_on_one_shoulder: a long springy pole resting on one shoulder
  - id=two_loads_at_pole_ends: two loads hanging one at each end of the pole
  - id=containers_at_both_ends: baskets, flat trays, wooden boxes or small glass cases
CONFUSABLE OBJECTS (must NOT be what is shown):
  - id=european_neck_yoke: a shaped yoke across both shoulders behind the neck
  - id=wheeled_food_cart: a pushed wheeled cart with a glass display case
  - id=head_carried_basket: a single basket balanced on top of the head
```

| mục | mô tả | có trong danh sách soi? | nguồn |
|---|---|---|---|
| `pole_on_one_shoulder` | a long springy pole resting on one shoulder | có | https://vi.wikipedia.org/wiki/G%C3%A1nh_h%C3%A… |
| `two_loads_at_pole_ends` | two loads hanging one at each end of the pole | có | https://vi.wikipedia.org/wiki/G%C3%A1nh_h%C3%A… |
| `containers_at_both_ends` | baskets, flat trays, wooden boxes or small glass cases | có | người Việt xác nhận 2026-09-18 |
| ~~`european_neck_yoke`~~ dễ nhầm | a shaped yoke across both shoulders behind the neck | — | European |
| ~~`wheeled_food_cart`~~ dễ nhầm | a pushed wheeled cart with a glass display case | — | generic/Thai |
| ~~`head_carried_basket`~~ dễ nhầm | a single basket balanced on top of the head | — | Indian/African |

---

## S003 · phở bò — *Vietnamese beef pho*
> prompt gốc: A steaming bowl of beef pho on a wooden table.

**Observer được chỉ soi:** `noodles, broth, beef`

**Critic đọc nguyên văn:**

```
VISUAL CONTRACT for Vietnamese beef pho
REQUIRED (each must be visible):
  - id=flat_rice_noodles: flat white rice noodles cut from thin sheets
  - id=clear_broth: clear thin broth, not milky or thickened
  - id=thin_sliced_beef: thin slices of beef laid over the noodles
  - id=scallion_and_onion_on_top: chopped scallion and thin raw onion on top
CONFUSABLE OBJECTS (must NOT be what is shown):
  - id=ramen: curly yellow wheat noodles with nori and halved egg
  - id=lanzhou_beef_noodle: round hand-pulled wheat noodles under red chili oil
  - id=udon: thick round white wheat noodles in pale broth
```

| mục | mô tả | có trong danh sách soi? | nguồn |
|---|---|---|---|
| `flat_rice_noodles` | flat white rice noodles cut from thin sheets | có | https://vi.wikipedia.org/wiki/Ph%E1%BB%9F |
| `clear_broth` | clear thin broth, not milky or thickened | có | https://vi.wikipedia.org/wiki/Ph%E1%BB%9F |
| `thin_sliced_beef` | thin slices of beef laid over the noodles | có | https://vi.wikipedia.org/wiki/Ph%E1%BB%9F |
| `scallion_and_onion_on_top` | chopped scallion and thin raw onion on top | **KHÔNG** | https://vi.wikipedia.org/wiki/Ph%E1%BB%9F |
| ~~`ramen`~~ dễ nhầm | curly yellow wheat noodles with nori and halved egg | — | Japanese |
| ~~`lanzhou_beef_noodle`~~ dễ nhầm | round hand-pulled wheat noodles under red chili oil | — | Chinese |
| ~~`udon`~~ dễ nhầm | thick round white wheat noodles in pale broth | — | Japanese |

---

## S004 · bánh chưng — *Banh chung, Vietnamese square sticky rice cake*
> prompt gốc: Square banh chung wrapped in dong leaves tied with bamboo strips on a tray.

**Observer được chỉ soi:** `leaves, bamboo, grid, corners`

**Critic đọc nguyên văn:**

```
VISUAL CONTRACT for Banh chung, Vietnamese square sticky rice cake
REQUIRED (each must be visible):
  - id=square_flat_block: a squat square block, clearly not a cylinder
  - id=dong_leaf_wrapper: wrapped in broad dark green dong leaves
  - id=bamboo_strip_ties: tied with flat bamboo strips crossing in a grid
  - id=folded_leaf_corners: leaf folded into sharp flat corners on top
CONFUSABLE OBJECTS (must NOT be what is shown):
  - id=zongzi: a pyramid or cone bundle in bamboo leaves tied with string
  - id=banana_leaf_tamale: a soft oblong parcel in pale banana leaf
  - id=banh_tet: a long cylindrical log wrapped in banana leaves
```

| mục | mô tả | có trong danh sách soi? | nguồn |
|---|---|---|---|
| `square_flat_block` | a squat square block, clearly not a cylinder | **KHÔNG** | https://vi.wikipedia.org/wiki/B%C3%A1nh_ch%C6%… |
| `dong_leaf_wrapper` | wrapped in broad dark green dong leaves | có | https://vi.wikipedia.org/wiki/B%C3%A1nh_ch%C6%… |
| `bamboo_strip_ties` | tied with flat bamboo strips crossing in a grid | có | https://vi.wikipedia.org/wiki/B%C3%A1nh_ch%C6%… |
| `folded_leaf_corners` | leaf folded into sharp flat corners on top | có | **CHƯA CÓ NGUỒN** |
| ~~`zongzi`~~ dễ nhầm | a pyramid or cone bundle in bamboo leaves tied with string | — | Chinese |
| ~~`banana_leaf_tamale`~~ dễ nhầm | a soft oblong parcel in pale banana leaf | — | Latin American |
| ~~`banh_tet`~~ dễ nhầm | a long cylindrical log wrapped in banana leaves | — | Vietnamese (southern variant, NOT banh chung) |

---

## S006 · chùa Một Cột — *One Pillar Pagoda, Hanoi*
> prompt gốc: The One Pillar Pagoda over its pond in early morning mist.

**Observer được chỉ soi:** `pillar, roof, corners, pond, stairway`

**Critic đọc nguyên văn:**

```
VISUAL CONTRACT for One Pillar Pagoda, Hanoi
REQUIRED (each must be visible):
  - id=single_stone_pillar: the whole structure resting on one stone pillar
  - id=small_square_pavilion: one small square wooden pavilion about three metres wide
  - id=upturned_tile_roof: a tiled roof with four upturned curved corners
  - id=standing_in_pond: rising from the middle of a small pond
  - id=narrow_stone_stairway: a narrow stone stairway climbing to the pavilion door
CONFUSABLE OBJECTS (must NOT be what is shown):
  - id=multi_tier_pagoda: a tall tower of many stacked roof tiers
  - id=floating_torii_shrine: a red gate or shrine on many pillars over water
```

| mục | mô tả | có trong danh sách soi? | nguồn |
|---|---|---|---|
| `single_stone_pillar` | the whole structure resting on one stone pillar | có | https://vi.wikipedia.org/wiki/Ch%C3%B9a_M%E1%B… |
| `small_square_pavilion` | one small square wooden pavilion about three metres wide | **KHÔNG** | https://vi.wikipedia.org/wiki/Ch%C3%B9a_M%E1%B… |
| `upturned_tile_roof` | a tiled roof with four upturned curved corners | có | https://vi.wikipedia.org/wiki/Ch%C3%B9a_M%E1%B… |
| `standing_in_pond` | rising from the middle of a small pond | có | https://vi.wikipedia.org/wiki/Ch%C3%B9a_M%E1%B… |
| `narrow_stone_stairway` | a narrow stone stairway climbing to the pavilion door | có | https://vi.wikipedia.org/wiki/Ch%C3%B9a_M%E1%B… |
| ~~`multi_tier_pagoda`~~ dễ nhầm | a tall tower of many stacked roof tiers | — | Chinese/Japanese |
| ~~`floating_torii_shrine`~~ dễ nhầm | a red gate or shrine on many pillars over water | — | Japanese |

---

## S008 · đèn lồng Hội An (phố cổ Hội An về đêm) — *Hoi An ancient town at night with silk lanterns*
> prompt gốc: Hoi An ancient town at night, silk lanterns along ochre walls.

**Observer được chỉ soi:** `silk, bamboo, ribs, lanterns, colours`

**Critic đọc nguyên văn:**

```
VISUAL CONTRACT for Hoi An ancient town at night with silk lanterns
REQUIRED (each must be visible):
  - id=silk_over_bamboo_ribs: silk stretched over slender curved bamboo ribs
  - id=many_lantern_colours: lanterns glowing in many different colours, not only red
  - id=lit_from_inside: each lantern glowing with light from inside
CONFUSABLE OBJECTS (must NOT be what is shown):
  - id=chinese_palace_lantern: cylindrical red lantern with gold tassels and Chinese characters
  - id=japanese_chochin: white paper lantern with horizontal ribs and black kanji
  - id=sky_lantern: floating paper lanterns rising into the night sky
```

| mục | mô tả | có trong danh sách soi? | nguồn |
|---|---|---|---|
| `silk_over_bamboo_ribs` | silk stretched over slender curved bamboo ribs | có | http://vanhoanghethuat.vn/nghe-thuat-trang-tri… |
| `many_lantern_colours` | lanterns glowing in many different colours, not only red | có | http://vanhoanghethuat.vn/nghe-thuat-trang-tri… |
| `lit_from_inside` | each lantern glowing with light from inside | **KHÔNG** | hiển nhiên trong cảnh đêm; prompt nói rõ 'at n… |
| ~~`chinese_palace_lantern`~~ dễ nhầm | cylindrical red lantern with gold tassels and Chinese characters | — | Chinese |
| ~~`japanese_chochin`~~ dễ nhầm | white paper lantern with horizontal ribs and black kanji | — | Japanese |
| ~~`sky_lantern`~~ dễ nhầm | floating paper lanterns rising into the night sky | — | Thai/Taiwanese |

---

## S009 · đàn bầu — *Dan bau, Vietnamese monochord*
> prompt gốc: A musician playing the dan bau monochord on a small stage.

**Observer được chỉ soi:** `string`

**Critic đọc nguyên văn:**

```
VISUAL CONTRACT for Dan bau, Vietnamese monochord
REQUIRED (each must be visible):
  - id=single_string: exactly one string running along the instrument
  - id=flexible_rod_at_one_end: a curved flexible rod standing up at one end
  - id=gourd_cup_on_rod: a gourd-shaped cup mounted on that rod
  - id=long_narrow_soundbox: a long narrow flat soundbox about one metre
  - id=laid_flat_and_plucked: laid flat, plucked with a stick held in one hand
CONFUSABLE OBJECTS (must NOT be what is shown):
  - id=guzheng_or_koto: a wide zither with many strings and movable bridges
  - id=erhu: a small two-string bowed fiddle held upright on the lap
  - id=ektara: a one-string lute with round gourd body held upright
```

| mục | mô tả | có trong danh sách soi? | nguồn |
|---|---|---|---|
| `single_string` | exactly one string running along the instrument | có | https://vi.wikipedia.org/wiki/%C4%90%C3%A0n_b%… |
| `flexible_rod_at_one_end` | a curved flexible rod standing up at one end | **KHÔNG** | https://en.wikipedia.org/wiki/%C4%90%C3%A0n_b%… |
| `gourd_cup_on_rod` | a gourd-shaped cup mounted on that rod | **KHÔNG** | https://en.wikipedia.org/wiki/%C4%90%C3%A0n_b%… |
| `long_narrow_soundbox` | a long narrow flat soundbox about one metre | **KHÔNG** | https://vi.wikipedia.org/wiki/%C4%90%C3%A0n_b%… |
| `laid_flat_and_plucked` | laid flat, plucked with a stick held in one hand | **KHÔNG** | https://en.wikipedia.org/wiki/%C4%90%C3%A0n_b%… |
| ~~`guzheng_or_koto`~~ dễ nhầm | a wide zither with many strings and movable bridges | — | Chinese/Japanese |
| ~~`erhu`~~ dễ nhầm | a small two-string bowed fiddle held upright on the lap | — | Chinese |
| ~~`ektara`~~ dễ nhầm | a one-string lute with round gourd body held upright | — | Indian |

---

## S010 · múa rối nước — *Vietnamese water puppetry*
> prompt gốc: A water puppetry performance at a water pavilion, puppets on the water surface.

**Observer được chỉ soi:** `puppets, water, pond, strings`

**Critic đọc nguyên văn:**

```
VISUAL CONTRACT for Vietnamese water puppetry
REQUIRED (each must be visible):
  - id=puppets_on_water_surface: puppets standing half submerged on open water
  - id=water_pavilion_behind: a tiled-roof pavilion rising out of the pond behind
  - id=curtain_hides_operators: a hanging screen across the pavilion hiding the puppeteers
  - id=no_strings_above_puppets: no rods or strings visible above the puppets
  - id=lacquered_wooden_puppets: carved wooden puppets with glossy painted lacquer
CONFUSABLE OBJECTS (must NOT be what is shown):
  - id=bunraku: puppets on a dry stage held by visible black-robed handlers
  - id=wayang_kulit: flat leather shadow puppets against a lit screen
  - id=marionette: puppets hanging from strings held from above
```

| mục | mô tả | có trong danh sách soi? | nguồn |
|---|---|---|---|
| `puppets_on_water_surface` | puppets standing half submerged on open water | có | https://vi.wikipedia.org/wiki/M%C3%BAa_r%E1%BB… |
| `water_pavilion_behind` | a tiled-roof pavilion rising out of the pond behind | có | https://vi.wikipedia.org/wiki/M%C3%BAa_r%E1%BB… |
| `curtain_hides_operators` | a hanging screen across the pavilion hiding the puppeteers | **KHÔNG** | https://vi.wikipedia.org/wiki/M%C3%BAa_r%E1%BB… |
| `no_strings_above_puppets` | no rods or strings visible above the puppets | có | https://vi.wikipedia.org/wiki/M%C3%BAa_r%E1%BB… |
| `lacquered_wooden_puppets` | carved wooden puppets with glossy painted lacquer | có | https://vi.wikipedia.org/wiki/M%C3%BAa_r%E1%BB… |
| ~~`bunraku`~~ dễ nhầm | puppets on a dry stage held by visible black-robed handlers | — | Japanese |
| ~~`wayang_kulit`~~ dễ nhầm | flat leather shadow puppets against a lit screen | — | Indonesian |
| ~~`marionette`~~ dễ nhầm | puppets hanging from strings held from above | — | European |

---

## S012 · thuyền thúng — *Vietnamese basket boat*
> prompt gốc: A fisherman paddling a round basket boat off a central Vietnam beach at sunrise.

**Observer được chỉ soi:** `hull, bamboo, sides, bow, stern`

**Critic đọc nguyên văn:**

```
VISUAL CONTRACT for Vietnamese basket boat
REQUIRED (each must be visible):
  - id=round_hull: a circular bowl-shaped hull
  - id=woven_bamboo: visible woven bamboo strips forming the body
  - id=no_bow_or_stern: low curved sides without a pointed bow or stern
  - id=thick_rim_ring: a thick bamboo or wooden ring around the top edge
  - id=hand_paddled: moved by hand paddling, no sail and no motor
CONFUSABLE OBJECTS (must NOT be what is shown):
  - id=wooden_sampan: a long narrow wooden boat with pointed ends
  - id=coracle: a round hide-covered or tarred canvas bowl boat
  - id=inflatable_raft: a round inflatable rubber raft with air tubes
```

| mục | mô tả | có trong danh sách soi? | nguồn |
|---|---|---|---|
| `round_hull` | a circular bowl-shaped hull | có | https://vi.wikipedia.org/wiki/Thuy%E1%BB%81n_t… |
| `woven_bamboo` | visible woven bamboo strips forming the body | có | https://vi.wikipedia.org/wiki/Thuy%E1%BB%81n_t… |
| `no_bow_or_stern` | low curved sides without a pointed bow or stern | có | https://vi.wikipedia.org/wiki/Thuy%E1%BB%81n_t… |
| `thick_rim_ring` | a thick bamboo or wooden ring around the top edge | có | https://vi.wikipedia.org/wiki/Thuy%E1%BB%81n_t… |
| `hand_paddled` | moved by hand paddling, no sail and no motor | **KHÔNG** | https://vi.wikipedia.org/wiki/Thuy%E1%BB%81n_t… |
| ~~`wooden_sampan`~~ dễ nhầm | a long narrow wooden boat with pointed ends | — | generic/Chinese |
| ~~`coracle`~~ dễ nhầm | a round hide-covered or tarred canvas bowl boat | — | Welsh/Irish |
| ~~`inflatable_raft`~~ dễ nhầm | a round inflatable rubber raft with air tubes | — | generic/modern |

---

## S013 · xích lô — *Vietnamese cyclo*
> prompt gốc: A cyclo carrying a passenger along a street.

**Observer được chỉ soi:** `seat, pedals, wheels`

**Critic đọc nguyên văn:**

```
VISUAL CONTRACT for Vietnamese cyclo
REQUIRED (each must be visible):
  - id=passenger_seat_in_front: the passenger seat sits ahead of the driver
  - id=driver_pedals_behind: the driver pedals on a raised saddle at the back
  - id=three_wheels_two_in_front: three wheels, two of them under the front seat
  - id=pedal_powered_no_engine: bicycle pedals and chain, no engine
CONFUSABLE OBJECTS (must NOT be what is shown):
  - id=cycle_rickshaw: a passenger bench behind the cyclist
  - id=tuk_tuk_auto_rickshaw: a motorised three-wheeler with an enclosed driver cab
  - id=side_car_trishaw: passengers seated in a sidecar beside the cyclist
```

| mục | mô tả | có trong danh sách soi? | nguồn |
|---|---|---|---|
| `passenger_seat_in_front` | the passenger seat sits ahead of the driver | có | https://vi.wikipedia.org/wiki/X%C3%ADch_l%C3%B… |
| `driver_pedals_behind` | the driver pedals on a raised saddle at the back | có | https://vi.wikipedia.org/wiki/X%C3%ADch_l%C3%B… |
| `three_wheels_two_in_front` | three wheels, two of them under the front seat | có | https://vi.wikipedia.org/wiki/X%C3%ADch_l%C3%B… |
| `pedal_powered_no_engine` | bicycle pedals and chain, no engine | có | https://vi.wikipedia.org/wiki/X%C3%ADch_l%C3%B… |
| ~~`cycle_rickshaw`~~ dễ nhầm | a passenger bench behind the cyclist | — | Indian/Bangladeshi |
| ~~`tuk_tuk_auto_rickshaw`~~ dễ nhầm | a motorised three-wheeler with an enclosed driver cab | — | Thai/Indian |
| ~~`side_car_trishaw`~~ dễ nhầm | passengers seated in a sidecar beside the cyclist | — | Filipino/Singaporean |

---

## S015 · chợ nổi — *Mekong Delta floating market*
> prompt gốc: A floating market at dawn, fruit boats with sample poles.

**Observer được chỉ soi:** `bamboo, pole, bow, boats`

**Critic đọc nguyên văn:**

```
VISUAL CONTRACT for Mekong Delta floating market
REQUIRED (each must be visible):
  - id=bamboo_sample_pole: a tall bamboo pole on the bow hanging sample produce
  - id=produce_laden_wooden_boats: long wooden boats piled with fruit and vegetables
  - id=boats_clustered_on_open_river: boats crowded together on a wide open river
  - id=no_land_stalls: trading happens boat to boat, no stalls on shore
CONFUSABLE OBJECTS (must NOT be what is shown):
  - id=damnoen_saduak_market: small paddled canoes in a narrow canal lined with stalls
  - id=lok_baintan_market: small dugout canoes with women in wide flat hats
```

| mục | mô tả | có trong danh sách soi? | nguồn |
|---|---|---|---|
| `bamboo_sample_pole` | a tall bamboo pole on the bow hanging sample produce | có | https://en.wikipedia.org/wiki/C%C3%A1i_R%C4%83… |
| `produce_laden_wooden_boats` | long wooden boats piled with fruit and vegetables | có | https://vi.wikipedia.org/wiki/Ch%E1%BB%A3_n%E1… |
| `boats_clustered_on_open_river` | boats crowded together on a wide open river | có | https://vi.wikipedia.org/wiki/Ch%E1%BB%A3_n%E1… |
| `no_land_stalls` | trading happens boat to boat, no stalls on shore | **KHÔNG** | **CHƯA CÓ NGUỒN** |
| ~~`damnoen_saduak_market`~~ dễ nhầm | small paddled canoes in a narrow canal lined with stalls | — | Thai |
| ~~`lok_baintan_market`~~ dễ nhầm | small dugout canoes with women in wide flat hats | — | Indonesian |

---

## S017 · quan họ — *Quan ho Bac Ninh folk singers*
> prompt gốc: Quan ho singers standing and singing on a boat.

**Observer được chỉ soi:** `hats, headscarf, hat, colours`

**Critic đọc nguyên văn:**

```
VISUAL CONTRACT for Quan ho Bac Ninh folk singers
REQUIRED (each must be visible):
  - id=women_wide_flat_hats: women wearing very wide flat-brimmed round hats
  - id=women_headscarf: women wearing a dark folded headscarf under the hat
  - id=women_layered_tunics: women in several layered tunics of different colours
CONFUSABLE OBJECTS (must NOT be what is shown):
  - id=chinese_opera_performer: painted face with tall beaded headdress and water sleeves
  - id=geisha: white face makeup, kimono with obi, holding a folding fan
  - id=hanbok_boat_singer: short jacket over a wide bell-shaped skirt
```

| mục | mô tả | có trong danh sách soi? | nguồn |
|---|---|---|---|
| `women_wide_flat_hats` | women wearing very wide flat-brimmed round hats | có | https://ich.unesco.org/en/RL/quan-ho-bac-ninh-… |
| `women_headscarf` | women wearing a dark folded headscarf under the hat | có | https://ich.unesco.org/en/RL/quan-ho-bac-ninh-… |
| `women_layered_tunics` | women in several layered tunics of different colours | có | https://vi.wikipedia.org/wiki/Quan_h%E1%BB%8D |
| ~~`chinese_opera_performer`~~ dễ nhầm | painted face with tall beaded headdress and water sleeves | — | Chinese |
| ~~`geisha`~~ dễ nhầm | white face makeup, kimono with obi, holding a folding fan | — | Japanese |
| ~~`hanbok_boat_singer`~~ dễ nhầm | short jacket over a wide bell-shaped skirt | — | Korean |

---

## S018 · cồng chiêng Tây Nguyên — *Central Highlands gong ensemble*
> prompt gốc: A Central Highlands gong ensemble performing around a fire.

**Observer được chỉ soi:** `gongs, rim`

**Critic đọc nguyên văn:**

```
VISUAL CONTRACT for Central Highlands gong ensemble
REQUIRED (each must be visible):
  - id=hand_carried_bronze_gongs: flat bronze gongs each carried in one hand
  - id=one_gong_per_player: a row of players each holding one different-sized gong
  - id=gongs_not_on_stands: gongs held by the rim, not mounted on racks
  - id=struck_with_mallet_or_fist: struck with a short padded mallet or the bare fist
  - id=highland_woven_dress: dark woven cloth with red and white geometric bands
CONFUSABLE OBJECTS (must NOT be what is shown):
  - id=gamelan: rows of knobbed gongs mounted on carved wooden racks
  - id=hanging_chinese_gong: one large gong hanging in a tall wooden frame
  - id=african_drum_circle: players seated behind tall hand drums, no metal gongs
```

| mục | mô tả | có trong danh sách soi? | nguồn |
|---|---|---|---|
| `hand_carried_bronze_gongs` | flat bronze gongs each carried in one hand | có | https://ich.unesco.org/en/RL/space-of-gong-cul… |
| `one_gong_per_player` | a row of players each holding one different-sized gong | **KHÔNG** | https://ich.unesco.org/en/RL/space-of-gong-cul… |
| `gongs_not_on_stands` | gongs held by the rim, not mounted on racks | có | https://ich.unesco.org/en/RL/space-of-gong-cul… |
| `struck_with_mallet_or_fist` | struck with a short padded mallet or the bare fist | **KHÔNG** | https://vi.wikipedia.org/wiki/C%E1%BB%93ng_chi… |
| `highland_woven_dress` | dark woven cloth with red and white geometric bands | **KHÔNG** | **CHƯA CÓ NGUỒN** |
| ~~`gamelan`~~ dễ nhầm | rows of knobbed gongs mounted on carved wooden racks | — | Indonesian |
| ~~`hanging_chinese_gong`~~ dễ nhầm | one large gong hanging in a tall wooden frame | — | Chinese |
| ~~`african_drum_circle`~~ dễ nhầm | players seated behind tall hand drums, no metal gongs | — | West African |

---

## S020 · nón quai thao — *Non quai thao, northern Vietnamese flat hat*
> prompt gốc: A northern Vietnamese young woman wearing a flat-brimmed quai thao hat.

**Observer được chỉ soi:** `brim, rim, silk`

**Critic đọc nguyên văn:**

```
VISUAL CONTRACT for Non quai thao, northern Vietnamese flat hat
REQUIRED (each must be visible):
  - id=wide_flat_disc_brim: a very wide flat disc brim, no pointed top
  - id=shallow_downturned_rim: a shallow rim turned down around the outer edge
  - id=inner_head_ring: a small ring under the centre gripping the head
  - id=silk_tassel_cords: thick coloured silk tassel cords hanging from the brim
CONFUSABLE OBJECTS (must NOT be what is shown):
  - id=chinese_douli: a plain woven straw flat hat with no tassel cords
  - id=straw_boater: a stiff flat straw hat with a raised cylindrical crown
  - id=non_la: a pointed conical leaf hat with a thin cloth chinstrap
```

| mục | mô tả | có trong danh sách soi? | nguồn |
|---|---|---|---|
| `wide_flat_disc_brim` | a very wide flat disc brim, no pointed top | có | https://vi.wikipedia.org/wiki/N%C3%B3n_quai_th… |
| `shallow_downturned_rim` | a shallow rim turned down around the outer edge | có | https://vi.wikipedia.org/wiki/N%C3%B3n_quai_th… |
| `inner_head_ring` | a small ring under the centre gripping the head | **KHÔNG** | https://vi.wikipedia.org/wiki/N%C3%B3n_quai_th… |
| `silk_tassel_cords` | thick coloured silk tassel cords hanging from the brim | có | https://vi.wikipedia.org/wiki/N%C3%B3n_quai_th… |
| ~~`chinese_douli`~~ dễ nhầm | a plain woven straw flat hat with no tassel cords | — | Chinese/Japanese |
| ~~`straw_boater`~~ dễ nhầm | a stiff flat straw hat with a raised cylindrical crown | — | European |
| ~~`non_la`~~ dễ nhầm | a pointed conical leaf hat with a thin cloth chinstrap | — | Vietnamese (different hat, NOT non quai thao) |

---

## S030 · khăn rằn — *Khan ran, southern Vietnamese checkered scarf*
> prompt gốc: A woman in a checkered scarf sitting in a small sampan.

**Observer được chỉ soi:** `grid, colours`

**Critic đọc nguyên văn:**

```
VISUAL CONTRACT for Khan ran, southern Vietnamese checkered scarf
REQUIRED (each must be visible):
  - id=small_checkered_grid: a grid of small squares from crossing stripes
  - id=only_two_colours: only two alternating colours, any pair
  - id=long_narrow_rectangle: a long narrow rectangle, roughly one metre long
  - id=draped_on_neck_or_head: draped round the neck or tied over the head
CONFUSABLE OBJECTS (must NOT be what is shown):
  - id=keffiyeh: a large square checked headscarf held by a black cord ring
  - id=tartan_plaid: crossing stripes in three or more colours
  - id=gingham_picnic_cloth: a wide square red-white checked tablecloth
```

| mục | mô tả | có trong danh sách soi? | nguồn |
|---|---|---|---|
| `small_checkered_grid` | a grid of small squares from crossing stripes | có | https://vi.wikipedia.org/wiki/Kh%C4%83n_r%E1%B… |
| `only_two_colours` | only two alternating colours, any pair | có | https://vi.wikipedia.org/wiki/Kh%C4%83n_r%E1%B… |
| `long_narrow_rectangle` | a long narrow rectangle, roughly one metre long | **KHÔNG** | https://vi.wikipedia.org/wiki/Kh%C4%83n_r%E1%B… |
| `draped_on_neck_or_head` | draped round the neck or tied over the head | **KHÔNG** | https://vi.wikipedia.org/wiki/Kh%C4%83n_r%E1%B… |
| ~~`keffiyeh`~~ dễ nhầm | a large square checked headscarf held by a black cord ring | — | Arab/Palestinian |
| ~~`tartan_plaid`~~ dễ nhầm | crossing stripes in three or more colours | — | Scottish |
| ~~`gingham_picnic_cloth`~~ dễ nhầm | a wide square red-white checked tablecloth | — | generic/Western |

---

## S031 · áo tứ thân — *Ao tu than, northern Vietnamese four-panel dress*
> prompt gốc: A northern Vietnamese young woman in a four-panel dress with a sash.

**Observer được chỉ soi:** `sash, buttons`

**Critic đọc nguyên văn:**

```
VISUAL CONTRACT for Ao tu than, northern Vietnamese four-panel dress
REQUIRED (each must be visible):
  - id=open_front_two_flaps: an open front with two separate hanging flaps
  - id=sash_tied_at_waist: a long cloth sash tied round the waist
  - id=below_knee_length: the outer garment hanging to below the knee
  - id=no_front_buttons: no buttons or fastening down the centre front
CONFUSABLE OBJECTS (must NOT be what is shown):
  - id=hanfu: a robe wrapped with a crossed Y-shaped front and wide sleeves
  - id=kimono: a wrapped robe closed by a wide stiff obi sash
  - id=ao_dai: a fitted closed tunic over trousers, slit only at the hips
```

| mục | mô tả | có trong danh sách soi? | nguồn |
|---|---|---|---|
| `open_front_two_flaps` | an open front with two separate hanging flaps | **KHÔNG** | https://vi.wikipedia.org/wiki/%C3%81o_t%E1%BB%… |
| `sash_tied_at_waist` | a long cloth sash tied round the waist | có | https://vi.wikipedia.org/wiki/%C3%81o_t%E1%BB%… |
| `below_knee_length` | the outer garment hanging to below the knee | **KHÔNG** | https://vi.wikipedia.org/wiki/%C3%81o_t%E1%BB%… |
| `no_front_buttons` | no buttons or fastening down the centre front | có | https://vi.wikipedia.org/wiki/%C3%81o_t%E1%BB%… |
| ~~`hanfu`~~ dễ nhầm | a robe wrapped with a crossed Y-shaped front and wide sleeves | — | Chinese |
| ~~`kimono`~~ dễ nhầm | a wrapped robe closed by a wide stiff obi sash | — | Japanese |
| ~~`ao_dai`~~ dễ nhầm | a fitted closed tunic over trousers, slit only at the hips | — | Vietnamese (different garment, NOT ao tu than) |

---

16 thực thể · 65 mục required · **3 mục chưa có nguồn**
