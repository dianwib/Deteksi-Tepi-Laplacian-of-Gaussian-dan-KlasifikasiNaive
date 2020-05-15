from flask import Flask, render_template, request
import numpy as np
from PIL import Image
from naive_bayes import K_mean
import parse_data_LOG as log
from random import randint
import parse_data_LOG


app = Flask(__name__)
@app.route('/',methods=['GET', 'POST'])
@app.route('/index',methods=['GET', 'POST'])
def hello_world():
    mode=''
    if request.method == 'POST':

        #Init Class
        Kmean = K_mean()
        dataset = Kmean.dataset
        mode = request.form['mode']
        print(mode,"mode")
        if mode == "akurasi":

            test_size = (100 - int(request.form['komposisi'])) / 100
            # print(test_size)
            jumlah_data=len(dataset)
            jumlah_data_test=int(jumlah_data*test_size)
            # print(jumlah_data_test)
            jumlah_data_train=jumlah_data-jumlah_data_test
            temp_index_data = list(range(jumlah_data))
            #get random data test
            temp_index_data_test=[]
            while len(temp_index_data_test)!=jumlah_data_test:
                index_random=randint(0,len(dataset)-1)
                if index_random not in temp_index_data_test:
                    temp_index_data_test.append(index_random)

            temp_index_data_train=np.delete(temp_index_data,[temp_index_data_test],0).tolist()
            new_dataset_train=np.delete(dataset,[temp_index_data_test],0).tolist()

            # fit model
            model = Kmean.summarize_by_class(new_dataset_train)

            temp_index_img=[]
            temp_hasil_pred=[]
            benar=0
            salah=0
            for index in temp_index_data_test:
                # print(dataset[index])
                if index < len(parse_data_LOG.file_data_sehat):#data sehat
                    # print(index,"Sehat",dataset[index][4])
                    temp_index_img.append(parse_data_LOG.file_data_sehat[index])
                else:

                    index_baru=index-len(parse_data_LOG.file_data_sehat)
                    temp_index_img.append(parse_data_LOG.file_data_sakit[index_baru])
                    # print(index,index_baru, "Sakit", dataset[index][4])


                # define a new record
                row = dataset[index][0:4]
                kelas_asli=dataset[index][4]
                # print(index,row)
                # predict the label
                label = Kmean.predict(model, row)

                # print('Data= ', row, ' Predicted: ', label, '/',list(Kmean.dict_kelas.keys())[list(Kmean.dict_kelas.values()).index(label)])

                hasil_label_prediksi = list(Kmean.dict_kelas.keys())[list(Kmean.dict_kelas.values()).index(label)]
                temp_hasil_pred.append(hasil_label_prediksi)
                if label==kelas_asli:
                    benar+=1
                else:
                    salah+=1
            akurasi=(benar/(benar+salah)) * 100


            return render_template('Klasifikasi.html',mode=mode,len_data_test=jumlah_data_test,hasil_pred=temp_hasil_pred,akurasi=akurasi,jml_data_test=jumlah_data_test,jml_data_train=jumlah_data_train,index_data_test=temp_index_data_test,index_data_train=temp_index_data_train,temp_img=temp_index_img,benar=benar,salah=salah)

        # mode == "testing klasifikasi":
        else:
            pilih_citra = str(request.form['pilih_metode'])
            url_citra_test = 'static/assets/Klasifikasi/testing/' + str(pilih_citra) + '.jpg'
            print(url_citra_test)
            data_test = log.convert_rgb_to_greyscale(url_citra_test)
            rata_1, std_1 = log.LOG(url_citra_test)
            rata_2, std_2 = log.LOG_1(data_test)
            # define a new record
            row = [rata_1, std_1, rata_2, std_2]
            nilai_probabilitas_baru = {}


            # fit model
            model = Kmean.summarize_by_class(dataset)

            # predict the label
            label = Kmean.predict(model, row)

            print('Data= ', row, ' Predicted: ', label, '/',
                  list(Kmean.dict_kelas.keys())[list(Kmean.dict_kelas.values()).index(label)])

            hasil_label_prediksi = list(Kmean.dict_kelas.keys())[list(Kmean.dict_kelas.values()).index(label)]
            nilai_probabilitas = Kmean.probabilities
            nilai_probabilitas_baru = {}
            print(Kmean.dict_kelas)
            for key in Kmean.dict_kelas:
                value_temp = Kmean.dict_kelas[key]
                nilai_probabilitas_baru[key] = float(nilai_probabilitas[value_temp])

            # return render_template('Klasifikasi.html')

            return render_template('Klasifikasi.html',mode=mode, hasil_label=hasil_label_prediksi, img=pilih_citra,
                                   nilai_fitur=row, nilai_prob=nilai_probabilitas_baru)

    else:
        return render_template('Klasifikasi.html',mode=mode)


if __name__ == "__main__":
    app.run(host="127.0.0.1")
