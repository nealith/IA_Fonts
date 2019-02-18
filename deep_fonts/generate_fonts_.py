import random
import numpy
import theano
import model
import csv

m = model.Model(artificial_font=True)
m.try_load()
run_fn = m.get_run_fn()
W = m.get_font_embeddings()
cov = numpy.cov(W.T)




def generate_font():
    return numpy.random.multivariate_normal(mean=numpy.zeros(m.d), cov=cov)

def generate_input(n_fonts=5):
    fonts = [generate_font() for f in xrange(n_fonts)]
    for f in xrange(n_fonts):
        a, b = fonts[f], fonts[(f+1)%n_fonts]
        for p in numpy.linspace(0, 1, 10):
            print f, p
            batch_is = numpy.zeros((m.k, m.d), dtype=theano.config.floatX)
            batch_js = numpy.zeros((m.k,), dtype=numpy.int32)
            for z in xrange(m.k):
                batch_is[z] = a * (1-p) + b * p
                batch_js[z] = z

            yield batch_is, batch_js,f,p

def generate_for_f_and_p(f,p,output):
    print f, p
    batch_is = numpy.zeros((m.k, m.d), dtype=theano.config.floatX)
    batch_js = numpy.zeros((m.k,), dtype=numpy.int32)
    for z in xrange(m.k):
        batch_is[z] = a * (1-p) + b * p
        batch_js[z] = z

    img = model.draw_grid(run_fn(input_i, input_j))
    path = output+'.png'
    img.save(path)

def generate():

    print 'generating...'
    frame = 0
    with open('data.csv', 'w') as csvfile:
        fieldnames = ['path', 'f','p']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for input_i, input_j,f2,p in generate_input(100):
            img = model.draw_grid(run_fn(input_i, input_j))
            path = 'font_'+str(frame)+'.png'
            img.save(path)
            writer.writerow({'path': path, 'f': f2, 'p': p})
            frame += 1

generate()
