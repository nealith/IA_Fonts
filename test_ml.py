from ml import train, save, load, predict

#model = train('deep_fonts/data.csv')
#save(model,'test_model')

model = load('test_model')
nothing = 0
predict(model,'test.jpg',nothing)
