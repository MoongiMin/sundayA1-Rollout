# Current motor mapping and Home

R25 names are model coordinates. They do not always follow conventional anatomical pitch/roll names.
The address is (board, ID); IDs 1–5 are reused on the two arm buses. Hand ID 6 is excluded from the 22-joint policy.

|R25 joint|Board|ID|Sign|Home tick|Observed ticks|R25 Home (rad)|
|---|---|---:|---:|---:|---|---:|
|r_shoulder_roll|right_arm|1|-1|2110|957–3033|+0.000000|
|r_shoulder_pitch|right_arm|2|-1|1962|379–2039|+0.150000|
|r_elbow_yaw|right_arm|3|-1|1810|699–2957|+0.000000|
|r_elbow_pitch|right_arm|4|-1|1678|1266–3214|+1.000000|
|r_wrist_yaw|right_arm|5|+1|2205|1075–3349|+0.000000|
|l_shoulder_roll|left_arm|1|-1|1824|838–3042|+0.000000|
|l_shoulder_pitch|left_arm|2|+1|2045|1747–3671|+0.150000|
|l_elbow_yaw|left_arm|3|-1|1989|991–3144|+0.000000|
|l_elbow_pitch|left_arm|4|+1|2718|1104–3066|+1.000000|
|l_wrist_yaw|left_arm|5|+1|2159|974–3249|+0.000000|
|r_hip_roll|lower_body|8|-1|1988|1047–2933|+0.417069|
|r_hip_pitch|lower_body|7|-1|2090|1473–3460|+0.010979|
|r_thigh_yaw|lower_body|9|-1|2075|1506–2643|-0.278984|
|r_knee_roll|lower_body|10|+1|1876|1354–2870|+0.510023|
|r_ankle_roll|lower_body|11|-1|2203|1109–2511|+0.207162|
|r_ankle_pitch|lower_body|12|-1|1985|1061–2992|-0.056437|
|l_hip_roll|lower_body|20|-1|2138|1107–2951|-0.402972|
|l_hip_pitch|lower_body|19|-1|2002|730–2693|-0.010791|
|l_thigh_yaw|lower_body|21|-1|1906|1506–2583|+0.269004|
|l_knee_roll|lower_body|22|-1|2119|1283–2933|-0.463225|
|l_ankle_roll|lower_body|23|+1|2101|1522–3006|+0.170363|
|l_ankle_pitch|lower_body|24|-1|2145|997–3060|+0.053483|
