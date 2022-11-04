"""
When you add new language translation, you need to add the translations for each key element (angry, disgust, fear, happy, sad, suprise, netural)
with the corresponding language key. Please be careful about the English characters. I.e. Wutend is originally Wütend but since 'ü' is not in 
en alphabet we should change it to 'u'

Languages Added:
"en": English -- It's default language and no need to be added again. Program will read en values from keys of this dictionary
"tr": Turkish (Türkçe)
"de": German (Deutsch)
"""

emotions_dict = {
    "angry": {"tr": "Kizgin", "de": "Wutend"},
    "disgust": {"tr": "Igrenme", "de": "der Ekel"},
    "fear": {"tr": "Korku", "de": "Furcht"},
    "happy": {"tr": "Mutluluk", "de": "Glucklich"},
    "sad": {"tr": "Uzuntu", "de": "Traurig"},
    "surprise": {"tr": "Saskinlik", "de": "Uberraschung"},
    "neutral": {"tr": "Notr", "de": "Neutral"},
}
