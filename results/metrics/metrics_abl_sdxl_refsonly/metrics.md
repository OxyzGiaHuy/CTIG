# Metric độc lập trên các lô K/O/R
Evaluator: `Qwen/Qwen2.5-VL-7B-Instruct` (không phải Mistral) · VQAScore: `Qwen2.5-VL-7B P(Yes) (không phải clip-flant5)` theo P0 gốc · contract đóng băng `data/contracts_v2.json` · CI bootstrap theo prompt, 2000 lần.

## SDXL — 50 prompt

| | CAIRE ↑ | VQAScore ↑ | VCFS ↑ | CCR ↓ | NRG ↑ |
|---|---:|---:|---:|---:|---:|
| A | – | 0.720 [0.630, 0.810] | 52.9 [44.1, 61.9] | 20.8 [13.2, 28.5] | – |
| I0 | – | 0.610 [0.500, 0.720] | 55.6 [46.4, 64.6] | 15.3 [9.3, 21.3] | – |
| I1 | – | 0.710 [0.610, 0.820] | 72.3 [65.5, 78.4] | 14.2 [8.0, 20.7] | 43.2 [28.7, 57.4] |
| **Δ (I1−I0)** | – | 0.100 [-0.020, 0.230] | 16.7 [8.8, 24.5] | -1.2 [-7.2, 4.7] | – |

| I1 so với I0 | tỷ lệ |
|---|---:|
| NRG > 0 | 29/50 |
| NRG = 0 | 6/50 |
| NRG < 0 | 3/50 |
| NRG = N/A (I0 không thiếu gì) | 12/50 |
| correct no-op | 0/50 |

| prompt | VCFS I0→I1 | CCR I0→I1 | VQA I0→I1 | NRG | D0 (thiếu ở I0) |
|---|---|---|---|---:|---|
| S001 | 25→50 | 33→67 | 0.029→0.963 | 33 | long_front_back_panels, side_slits_at_hips, worn_over_trouse |
| S002 | 33→67 | 33→0 | 0.593→0.995 | 50 | pole_on_one_shoulder, two_suspended_loads |
| S003 | 73→73 | 0→0 | 0.999→0.998 | 0 | flat_rice_noodles |
| S004 | 38→62 | 33→100 | 0.893→0.971 | 40 | bamboo_strip_ties, square_parcel |
| S005 | 100→67 | 0→0 | 0.995→0.991 | N/A |  |
| S006 | 50→50 | 50→50 | 0.986→0.026 | 0 | single_central_stone_pillar, small_square_pavilion |
| S007 | 0→33 | 0→0 | 0.202→0.321 | 33 | doctoral_stelae, khue_van_cac_pavilion, stelae_on_tortoises |
| S008 | 100→100 | 0→0 | 0.990→0.999 | N/A |  |
| S009 | 0→25 | 33→33 | 0.023→0.469 | 25 | flexible_pitch_rod, long_narrow_resonator, plucked_with_plec |
| S010 | 50→100 | 0→0 | 0.940→0.999 | 100 | water_pavilion |
| S011 | 33→100 | 50→0 | 0.777→0.408 | 100 | live_stage_performer, spoken_sung_drama |
| S012 | 25→25 | 33→33 | 0.967→0.012 | -67 | person_paddling, reinforced_round_rim, woven_bamboo_body |
| S013 | 67→33 | 33→33 | 0.999→0.933 | -50 | driver_pedals_behind |
| S014 | 100→100 | 0→0 | 0.933→0.994 | N/A |  |
| S015 | 67→67 | 17→17 | 0.148→0.068 | 0 | sample_goods_on_poles |
| S016 | 67→83 | 0→0 | 0.881→0.915 | 50 | village_central_location |
| S017 | 0→22 | 0→0 | 0.029→0.963 | 22 | female_hat_and_headscarf, female_layered_attire, male_female |
| S018 | 0→67 | 0→0 | 0.881→0.001 | 67 | bronze_circular_gongs, ensemble_performance, struck_with_mal |
| S019 | 33→100 | 0→0 | 0.977→0.977 | 100 | dark_long_tunic, red_textile_decoration |
| S020 | 33→67 | 33→33 | 0.006→0.033 | 50 | decorative_silk_cords, very_wide_flat_crown |
| S021 | 67→100 | 0→0 | 0.755→0.992 | 100 | lanterns_carried_on_sticks |
| S022 | 67→67 | 0→0 | 0.223→0.223 | 0 | ong_dia_with_fan |
| S023 | 0→71 | 0→0 | 0.000→0.777 | 71 | kumquat_and_red_envelopes, shared_vietnamese_feast, tet_food |
| S024 | 0→75 | 50→0 | 0.001→0.500 | 75 | coffee_dripping, condensed_milk_and_ice, metal_phin_on_glass |
| S025 | 100→100 | 0→0 | 0.407→0.133 | N/A |  |
| S026 | 100→100 | 50→0 | 0.037→0.989 | N/A |  |
| S027 | 50→50 | 0→0 | 0.321→0.623 | 0 | thick_cylindrical_rice_noodles |
| S028 | 25→62 | 0→0 | 0.915→0.992 | 50 | cylindrical_log, sticky_rice_with_mung_bean_filling |
| S029 | 100→50 | 50→0 | 0.836→0.095 | N/A |  |
| S030 | 100→100 | 67→33 | 0.980→0.818 | N/A |  |
| S031 | 50→75 | 0→33 | 0.076→0.836 | 50 | four_long_panels, open_unbuttoned_front |
| S032 | 67→67 | 50→25 | 0.378→0.593 | 0 | finger_picks |
| S033 | 25→62 | 0→0 | 0.408→0.993 | 50 | court_music_ensemble, mixed_traditional_instruments |
| S034 | 100→75 | 0→0 | 0.321→0.245 | N/A |  |
| S035 | 64→29 | 0→0 | 0.967→0.893 | -80 | altar_context, ritual_text_reading |
| S036 | 0→50 | 0→0 | 0.679→0.993 | 50 | bride_and_groom, multiple_red_gift_trays |
| S037 | 75→100 | 0→0 | 0.778→0.998 | 100 | coffee_service |
| S038 | 100→67 | 0→0 | 0.999→0.963 | N/A |  |
| S039 | 67→100 | 0→0 | 0.992→0.998 | 100 | main_and_false_doors |
| S040 | 50→62 | 50→50 | 0.996→0.915 | 20 | chin_strap, leaf_covering_on_bamboo_frame |
| S041 | 71→100 | 0→0 | 0.005→0.974 | 100 | peach_blossoms_for_sale |
| S042 | 100→100 | 50→50 | 0.999→0.999 | N/A |  |
| S043 | 100→75 | 0→0 | 0.993→0.995 | N/A |  |
| S044 | 71→86 | 0→50 | 0.623→0.947 | 50 | tet_context |
| S045 | 75→100 | 0→50 | 0.755→0.940 | 100 | five_fruit_tray |
| S046 | 33→67 | 0→0 | 0.020→0.007 | 50 | minimal_or_single_handrail, very_narrow_bamboo_crossing |
| S047 | 33→67 | 0→0 | 0.924→0.999 | 50 | dense_textile_decoration, pleated_skirt_or_group_specific_lo |
| S048 | 100→100 | 0→0 | 0.999→1.000 | N/A |  |
| S049 | 33→67 | 50→50 | 0.706→0.202 | 50 | dark_long_male_tunic, standing_collar_and_side_fastening |
| S050 | 62→100 | 0→0 | 0.269→0.997 | 100 | adult_gives_child |
