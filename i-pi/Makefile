# Makefile for the CP2K example
#
# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

.PHONY: all clean h2o
all: h2o

DRIVER:=pw.x
IPI:=i-pi

-include make.in

define run_driver
  for i in `seq 1 $1`; do \
    mkdir -p run_$$i; cd run_$$i; \
    $(DRIVER) < ../$2 > pw.out --ipi $3 & cd ..; \
	sleep 3; \
  done;
endef

h2o:
	cd h2o; $(IPI) input.xml & sleep 3; \
	$(call run_driver,4,pw.in,localhost:31415) \
	wait
clean:
	rm -rf */\#*\#* */*simulation* */*.out* */*.log* */*run_* */*RESTART* */EXIT
