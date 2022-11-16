#! /bin/csh

set data_path = "./data"

set data = "${data_path}/coked_mo_control.txt"
set data = "${data_path}/coked_mo_2step.txt"

python mlp.py ${data}

