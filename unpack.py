import pickle

with open("data.pkl", "rb") as pickle_in:
  data = pickle.load(pickle_in)

  print("hotdog")
  print(data)
