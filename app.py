import pickle

with open('gender_model.pkl', 'rb') as f:
    model = pickle.load(f)

name = input('Enter a name: ')
prediction = model.predict([name])[0]

print(f'Predicted Gender: {prediction}')
