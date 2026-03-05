import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("new_data.csv")
print(data.head())

print(data.columns.tolist())

print(data.describe())

data.hist("emotion_intensity", by="painting")
plt.suptitle("Emotion Intensity")
plt.show()

data.hist("feel_sombre", by="painting")
plt.suptitle("Feel Sombre")
plt.show()

data.hist("feel_content", by="painting")
plt.suptitle("Feel Content")
plt.show()

data.hist("feel_calm", by="painting")
plt.suptitle("Feel Calm")
plt.show()

data.hist("feel_uneasy", by="painting")
plt.suptitle("Feel Uneasy")
plt.show()

data.hist("prominent_colours", by="painting")
plt.suptitle("Prominent Colours")
plt.show()

data.hist("object_noticed", by="painting")
plt.suptitle("Object Noticed")
plt.show()

data.hist("willingness_to_pay", by="painting")
plt.suptitle("Willingness to Pay")
plt.show()
