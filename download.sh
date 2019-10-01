#!/bin/bash

enunciado_lnk=https://campus.exactas.uba.ar/pluginfile.php/143556/course/section/19842/tp2.pdf
data_lnk=https://campus.exactas.uba.ar/pluginfile.php/143556/course/section/19842/imdb.tar.gz
data_doc_lnk=https://campus.exactas.uba.ar/pluginfile.php/143556/course/section/19842/dataset.pdf
book_lnk=http://93.174.95.29/main/1610000/5f88a9f135b7ab31fbcf1729412560dc/%28Springer%20Series%20in%20Statistics%29%20Trevor%20Hastie%2C%20%20Robert%20Tibshirani%2C%20Jerome%20Friedman%20-%20The%20Elements%20of%20%20Statistical%20Learning_%20%20Data%20Mining%2C%20Inference%2C%20and%20Prediction.-Springer%20%282013%29.pdf

function urldecode(){
    # años usando bash y hoy aprendí algo nuevo, man bash, buscá el builtin ':', y el parámetro especial '_'
    : "${*//+/}"; echo -e "${_//%/\\x}";
}
function download_in(){
    link=$1; shift; dir=$1; shift; file=${*} # 3 args
    if [ ! -f "${dir}/${file}" ]
    then
        echo "descargando ${file} en ${dir} ... "
        wget $link -P $dir
    else
        echo "${file} ya descargado!, no hago nada ..."
    fi

}

echo "descargando data"
file=$(basename $data_lnk)
download_in $data_lnk data file
echo "descomprimir data con tar -zxf data/${file}"

[ ! -d doc ] && mkdir doc
echo "descargando documentos:"
download_in $enunciado_lnk doc $(basename $enunciado_lnk)
download_in $data_doc_lnk doc $(basename $data_doc_lnk)
download_in $book_lnk doc $(echo $(urldecode $book_lnk) | sed 's:.*/::')
