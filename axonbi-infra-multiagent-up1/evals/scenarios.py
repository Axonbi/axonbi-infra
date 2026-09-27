"""
Scripted patient conversations for measuring tokens and calls per turn.

The patient side is fixed, so the baseline and the new architecture are
measured on exactly the same messages. Written in the tenant's (Saudi)
dialect, from the channel number that owns the seeded booking in
tests/fake_hospital.py, so cancel/reschedule have something to find and
no OTP is needed.
"""

CHANNEL_PHONE = "+966500000001"

SCENARIOS = {
    "faq": [
        "السلام عليكم",
        "وش أوقات الدوام عندكم؟",
        "طيب وين موقع فرع النزهة؟",
        "شكرا",
    ],
    "booking": [
        "ابي احجز موعد جلدية",
        "1",
        "بكرة",
        "2",
        "ايه نفس الرقم",
        "محمد أحمد العتيبي",
        "ايوه أكد",
    ],
    "medical": [
        "عندي حكة في يدي من ثلاث أيام",
        "لا ما فيه حرارة بس تحكني كثير",
        "ايه احجز لي",
    ],
    "cancel": [
        "ابي الغي موعدي",
        "نفس رقم الواتساب",
        "ايوه الغيه",
    ],
    "reschedule": [
        "ابي اغير موعدي",
        "نفس الرقم",
        "بكرة",
        "1",
        "ايوه",
    ],
    "complaint": [
        "ابي اقدم شكوى",
        "الاستقبال في فرع النزهة خلوني انتظر ساعة كاملة",
        "محمد العتيبي",
        "ايوه ارسلها",
    ],
}
