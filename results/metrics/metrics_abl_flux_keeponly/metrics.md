# Metric độc lập trên các lô K/O/R
Evaluator: `Qwen/Qwen2.5-VL-7B-Instruct` (không phải Mistral) · VQAScore: `Qwen2.5-VL-7B P(Yes) (không phải clip-flant5)` theo P0 gốc · contract đóng băng `data/contracts_v2.json` · CI bootstrap theo prompt, 2000 lần.

## FLUX.1-dev — 50 prompt

| | CAIRE ↑ | VQAScore ↑ | VCFS ↑ | CCR ↓ | NRG ↑ |
|---|---:|---:|---:|---:|---:|
| A | – | 0.670 [0.570, 0.760] | 44.8 [35.0, 54.4] | 13.3 [7.3, 19.7] | – |
| I0 | – | 0.700 [0.600, 0.780] | 59.1 [50.4, 67.1] | 15.0 [8.3, 22.3] | – |
| I1 | – | 0.710 [0.610, 0.800] | 66.1 [57.5, 74.0] | 17.2 [11.3, 23.3] | 20.7 [4.6, 36.2] |
| **Δ (I1−I0)** | – | 0.010 [-0.060, 0.090] | 7.1 [0.1, 14.2] | 2.2 [-2.7, 7.2] | – |

| I1 so với I0 | tỷ lệ |
|---|---:|
| NRG > 0 | 22/50 |
| NRG = 0 | 13/50 |
| NRG < 0 | 5/50 |
| NRG = N/A (I0 không thiếu gì) | 10/50 |
| correct no-op | 0/50 |

| prompt | VCFS I0→I1 | CCR I0→I1 | VQA I0→I1 | NRG | D0 (thiếu ở I0) |
|---|---|---|---|---:|---|
| S001 | 0→25 | 33→33 | 0.991→0.995 | 25 | fitted_long_sleeved_bodice, long_front_back_panels, side_sli |
| S002 | 67→100 | 0→0 | 0.999→0.999 | 100 | pole_on_one_shoulder |
| S003 | 0→45 | 0→17 | 0.562→0.990 | 45 | clear_broth, flat_rice_noodles, onion_garnish, thin_sliced_b |
| S004 | 62→100 | 100→67 | 0.940→0.990 | 100 | square_parcel |
| S005 | 100→100 | 0→0 | 0.947→0.852 | N/A |  |
| S006 | 50→50 | 50→50 | 0.321→0.321 | 0 | single_central_stone_pillar, small_square_pavilion |
| S007 | 33→0 | 0→0 | 0.068→0.269 | -100 | doctoral_stelae, khue_van_cac_pavilion |
| S008 | 100→100 | 0→0 | 0.998→1.000 | N/A |  |
| S009 | 25→25 | 33→33 | 0.321→0.202 | 0 | flexible_pitch_rod, plucked_with_plectrum, single_string |
| S010 | 100→100 | 0→0 | 0.999→0.798 | N/A |  |
| S011 | 33→67 | 0→0 | 0.623→0.562 | 50 | spoken_sung_drama, stage_scenery |
| S012 | 88→75 | 33→33 | 0.562→0.623 | 0 | woven_bamboo_body |
| S013 | 33→50 | 33→17 | 0.993→0.963 | 25 | driver_pedals_behind, passenger_seat_in_front |
| S014 | 100→100 | 0→0 | 0.881→0.915 | N/A |  |
| S015 | 67→67 | 17→17 | 0.988→0.532 | 0 | sample_goods_on_poles |
| S016 | 83→83 | 0→50 | 0.867→0.778 | 0 | village_central_location |
| S017 | 33→0 | 0→0 | 0.990→0.623 | -100 | female_hat_and_headscarf, female_layered_attire, male_tradit |
| S018 | 0→0 | 33→0 | 0.294→0.378 | 0 | bronze_circular_gongs, ensemble_performance, struck_with_mal |
| S019 | 33→67 | 0→0 | 0.940→0.852 | 50 | dark_long_tunic, red_textile_decoration |
| S020 | 67→0 | 33→33 | 0.008→0.006 | -100 | very_wide_flat_crown |
| S021 | 100→100 | 0→0 | 0.994→0.997 | N/A |  |
| S022 | 67→67 | 0→0 | 0.349→0.438 | 0 | ong_dia_with_fan |
| S023 | 29→57 | 0→0 | 0.706→0.321 | 40 | shared_vietnamese_feast, tet_food_marker |
| S024 | 75→62 | 0→0 | 0.623→0.053 | 50 | condensed_milk_and_ice |
| S025 | 75→88 | 0→0 | 0.321→0.731 | 50 | peanut_hoisin_dip |
| S026 | 100→100 | 50→50 | 1.000→0.995 | N/A |  |
| S027 | 50→50 | 0→0 | 0.563→0.593 | 0 | beef_and_spicy_broth |
| S028 | 62→62 | 0→0 | 0.818→0.986 | 40 | sticky_rice_with_mung_bean_filling |
| S029 | 50→100 | 50→50 | 0.852→0.940 | 100 | lotus_leaf_wrapper |
| S030 | 75→75 | 33→33 | 0.562→1.000 | 0 | worn_on_head_or_neck |
| S031 | 25→25 | 0→0 | 0.119→0.037 | 0 | four_long_panels, open_unbuttoned_front, visible_yem_and_inn |
| S032 | 33→67 | 0→50 | 0.004→0.060 | 50 | finger_picks, long_horizontal_zither |
| S033 | 25→81 | 0→0 | 0.755→0.982 | 75 | court_music_ensemble, mixed_traditional_instruments |
| S034 | 75→62 | 0→0 | 0.060→0.707 | 50 | courtyard_and_banyan |
| S035 | 57→14 | 0→0 | 0.893→0.947 | -100 | altar_context, ao_the_khan_xep_prompt_attire |
| S036 | 50→100 | 0→0 | 0.182→0.993 | 100 | multiple_red_gift_trays |
| S037 | 100→100 | 0→0 | 0.997→0.997 | N/A |  |
| S038 | 33→67 | 0→0 | 1.000→1.000 | 50 | low_straight_bunds, shallow_evaporation_pans |
| S039 | 67→67 | 0→50 | 0.940→0.990 | 0 | red_brick_tower |
| S040 | 75→75 | 50→50 | 0.998→0.998 | 0 | chin_strap |
| S041 | 71→50 | 0→0 | 0.269→0.562 | 30 | peach_blossoms_for_sale |
| S042 | 100→100 | 100→50 | 0.953→0.999 | N/A |  |
| S043 | 50→100 | 0→0 | 0.679→0.933 | 100 | woodblock_printing_action |
| S044 | 86→71 | 50→50 | 0.798→0.623 | 0 | tet_context |
| S045 | 81→62 | 0→50 | 0.915→0.076 | -10 | raised_ancestral_altar |
| S046 | 33→67 | 50→50 | 0.245→0.003 | 50 | minimal_or_single_handrail, very_narrow_bamboo_crossing |
| S047 | 33→50 | 0→0 | 0.999→0.993 | 25 | dense_textile_decoration, pleated_skirt_or_group_specific_lo |
| S048 | 100→100 | 0→0 | 0.998→0.999 | N/A |  |
| S049 | 0→33 | 0→25 | 0.933→0.933 | 33 | dark_long_male_tunic, folded_wrapped_headcloth, standing_col |
| S050 | 100→100 | 0→0 | 0.980→0.991 | N/A |  |
