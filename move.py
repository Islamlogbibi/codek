import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def is_hand_closed(hand_landmarks):
    # بنقارن بين طرف الاصبع الوسطى ومفصلها القريب من الكف
    # لو الطرف تحت المفصل، معناها الأصبع مطوية -> اليد مغلقة
    tip_ids = [8, 12, 16, 20]  # أطراف الأصابع ما عدا الإبهام
    closed_fingers = 0
    for tip_id in tip_ids:
        if hand_landmarks.landmark[tip_id].y > hand_landmarks.landmark[tip_id - 2].y:
            closed_fingers += 1
    return closed_fingers >= 3  # نعتبرها مغلقة لو 3 أصابع أو أكثر مطوية

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)  # قلب الصورة أفقيًا كأنها مرآة
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        left_status = right_status = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label  # 'Left' or 'Right'
                closed = is_hand_closed(hand_landmarks)
                if label == 'Left':
                    left_status = 'closed' if closed else 'open'
                else:
                    right_status = 'closed' if closed else 'open'

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # التحقق من الحالات الأربعة
            if left_status == 'closed' and right_status == 'open':
                print('right')
            elif left_status == 'open' and right_status == 'closed':
                print('left')
            elif left_status == 'closed' and right_status == 'closed':
                print('top')
            elif left_status == 'open' and right_status == 'open':
                print('bottom')

        cv2.imshow('Hand Detection', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
