#!/bin/bash
# auto_train.sh
# train.pyをいろいろな条件で試したい時のスクリプト
# train.pyの引数を手入力するため、ミスが発生しやすい。
# auto_train.shを修正したら、一度-cオプションを実行してミスがないか確認するべき

# オプション引数を判定する部分（変更しない）

usage_exit() {
    echo "Usage: $0 [-c]" 1>&2
    echo " -c: 設定が正常に動作するか確認する"
    exit 1
}

FLAG_CHK=""
while getopts ch OPT
do
    case $OPT in
        c)  FLAG_CHK="--only_check"
            ;;
        h)  usage_exit
            ;;
        \?) usage_exit
            ;;
    esac
done

shift $((OPTIND - 1))

# 以下自由に変更する部分（オプション引数を反映させるなら、$FLG_CHKは必要）

COUNT=1
echo -e "\n<< test ["${COUNT}"] >>\n"
./create_dataset.py -in 2000 -t 0.95
./train.py -u64 -b10 -g0 -r result/002/*_200* -o result/002 -e300 $FLAG_CHK

COUNT=$(( COUNT + 1 ))
echo -e "\n<< test ["${COUNT}"] >>\n"
./create_dataset.py -in 2000 -t 0.95
./train.py -u64 -b10 -g0 -r result/002/*_300* -o result/002 -e400 $FLAG_CHK

COUNT=$(( COUNT + 1 ))
echo -e "\n<< test ["${COUNT}"] >>\n"
./create_dataset.py -in 2000 -t 0.95
./train.py -u64 -b10 -g0 -r result/002/*_400* -o result/002 -e500 $FLAG_CHK

COUNT=$(( COUNT + 1 ))
echo -e "\n<< test ["${COUNT}"] >>\n"
./create_dataset.py -in 2000 -t 0.95
./train.py -u64 -b10 -g0 -r result/002/*_500* -o result/002 -e600 $FLAG_CHK

COUNT=$(( COUNT + 1 ))
echo -e "\n<< test ["${COUNT}"] >>\n"
./create_dataset.py -in 2000 -t 0.95
./train.py -u64 -b10 -g0 -r result/002/*_600* -o result/002 -e700 $FLAG_CHK
