mkdir datasets/test/$1_$2
cp results/$1/test_$2/images/*_fake_A.png datasets/test/$1_$2
cd datasets/test/$1_$2
for var in *.png; do mv "$var" "${var%_fake_A.png}.png"; done
cd ..
python test.py $1_$2