.PHONY: all learnfunc1 learnfunc2 learnhans2 learnfunc3 learnhans3 clean normalize rvg clean1 clean2 clean3 show1 show2 show3




all: learnfunc1 learnhans2 learnhans3





learnfunc1: learnfunc1.done
learnfunc1.done: enc.ctx gen1.ctx src1.rgb tgt1.rgb
	tewel learnfunc enc.ctx gen1.ctx src=src1.rgb tgt=tgt1.rgb dim=128x128 mul=1 reps=32768 verbose=1
	tewel learnfunc enc.ctx gen1.ctx src=src1.rgb tgt=tgt1.rgb dim=128x128 mul=0.25 reps=4096 verbose=1
	touch $@

tgt1.rgb: iden.ctx
	tewel synth iden.ctx verbose=1 dim=128x128 out=$@ src=/spin/dan/lq.128

src1.rgb: tgt1.rgb
	cp -f $< $@










learnfunc2: learnfunc2.done
learnfunc2.done: src2.rgb tgt2.rgb gen2.ctx pgen2.ctx
	tewel learnfunc gen2.ctx pgen2.ctx src=src2.rgb tgt=tgt2.rgb srcdim=32x32x3 dim=32x32x3 tgtdim=32x32x12 reps=256
	touch $@

learnhans2: learnhans2.done
learnhans2.done: learnfunc2.done src2.rgb tgt2.rgb gen2.ctx dis2.ctx pgen2.ctx
	tewel learnhans gen2.ctx pgen2.ctx src=src2.rgb tgt=tgt2.rgb dis=dis2.ctx dim=32x32x3 srcdim=32x32x3 tgtdim=32x32x12 noise=0.0 reps=32768
	tewel learnhans gen2.ctx pgen2.ctx src=src2.rgb tgt=tgt2.rgb dis=dis2.ctx dim=32x32x3 srcdim=32x32x3 tgtdim=32x32x12 noise=0.0 reps=4096 mul=0.25
	touch $@

src2.rgb: dns1.ctx mean.ctx enc.ctx gen1.ctx iden.ctx learnfunc1.done
	rm -f src2.tmp.rgb
	tewel synth dns1.ctx mean.ctx enc.ctx gen1.ctx \
		dim=256x256x3 verbose=1 src=/spin/dan/hq.256 out=src2.tmp.rgb
	tewel synth iden.ctx \
		randseed=1 verbose=1 lowmem=1 srctrav=rand crop=random \
		srcdim=128x128x3 dim=32x32x3 limit=65536 src=src2.tmp.rgb out=$@
	rm -f src2.tmp.rgb

tgt2.rgb: dns1.ctx iden.ctx ups1.ctx
	rm -f tgt2.tmp.rgb
	tewel synth dns1.ctx \
		dim=256x256x3 verbose=1 src=/spin/dan/hq.256 out=tgt2.tmp.rgb
	tewel synth ups1.ctx iden.ctx dns1.ctx \
		randseed=1 verbose=1 lowmem=1 srctrav=rand crop=random \
		srcdim=128x128x12 dim=32x32x12 limit=65536 src=tgt2.tmp.rgb out=$@
	rm -f tgt2.tmp.rgb








learnfunc3: learnfunc3.done
learnfunc3.done: gen3.ctx pgen3.ctx src3.rgb tgt3.rgb
	tewel learnfunc gen3.ctx pgen3.ctx src=src3.rgb tgt=tgt3.rgb srcdim=32x32x3 dim=32x32x3 tgtdim=32x32x12 reps=256
	touch $@

learnhans3: learnhans3.done
learnhans3.done: learnfunc3.done gen3.ctx dis3.ctx pgen3.ctx src3.rgb tgt3.rgb
	tewel learnhans gen3.ctx pgen3.ctx src=src3.rgb tgt=tgt3.rgb dis=dis3.ctx dim=32x32x3 srcdim=32x32x3 tgtdim=32x32x12 noise=0.5 reps=32768
	tewel learnhans gen3.ctx pgen3.ctx src=src3.rgb tgt=tgt3.rgb dis=dis3.ctx dim=32x32x3 srcdim=32x32x3 tgtdim=32x32x12 noise=0.5 reps=4096 mul=0.25
	touch $@

src3.rgb: dns1.ctx mean.ctx
	tewel synth dns1.ctx mean.ctx \
		randseed=1 verbose=1 lowmem=1 srctrav=rand crop=random \
		dim=64x64x3 limit=65536 src=/spin/dan/dne.512 out=$@

tgt3.rgb: dns1.ctx
	tewel synth dns1.ctx \
		randseed=1 verbose=1 lowmem=1 srctrav=rand crop=random \
		dim=64x64x3 limit=65536 src=/spin/dan/dne.512 out=$@



%.ctx: %.txt
	tewel make $@ spec=$< clobber=1





normalize: nrm.ctx den.ctx

rvg: src1.rvg shampane.rvg nova2.rvg miniceleb.rvg lfw.rvg minidne.rvg cand.rvg

nrm.ctx:
	tewel identity nrm.ctx dim=1536 clobber=1
den.ctx: nrm.ctx src1.rvg
	tewel identity den.ctx dim=1536 clobber=1
	tewel normalize nrm.ctx den.ctx rvg=src1.rvg




src1.rvg: src1.rgb enc.ctx
	tewel synth enc.ctx dim=128x128 src=$< out=$@ verbose=1

shampane.rvg: dns1.ctx mean.ctx enc.ctx
	tewel synth dns1.ctx mean.ctx enc.ctx dim=256x256 src=/spin/dan/shampane.aligned out=$@ verbose=1
nova2.rvg: dns1.ctx mean.ctx enc.ctx
	tewel synth dns1.ctx mean.ctx enc.ctx dim=256x256 src=/spin/dan/nova2.aligned out=$@ verbose=1
miniceleb.rvg: dns1.ctx mean.ctx enc.ctx
	tewel synth dns1.ctx mean.ctx enc.ctx dim=256x256 src=/spin/dan/miniceleb.aligned out=$@ verbose=1
lfw.rvg: dns1.ctx mean.ctx enc.ctx
	tewel synth dns1.ctx mean.ctx enc.ctx dim=256x256 src=/spin/dan/lfw.aligned out=$@ verbose=1
minidne.rvg: dns1.ctx mean.ctx enc.ctx
	tewel synth dns1.ctx mean.ctx enc.ctx dim=256x256 src=/spin/dan/minidne.aligned out=$@ verbose=1
cand.rvg: dns1.ctx mean.ctx enc.ctx
	tewel synth dns1.ctx mean.ctx enc.ctx dim=256x256 src=/spin/dan/cand.aligned out=$@ verbose=1






show1: enc.ctx gen1.ctx
	tewel synth enc.ctx gen1.ctx src=/spin/dan/lq.128 \
		srctrav=rand dim=128x128 repeat=1 reload=1 delay=1 outkind=sdl
show2: enc.ctx gen1.ctx gen2.ctx
	tewel synth enc.ctx gen1.ctx gen2.ctx src=/spin/dan/lq.128 \
		srctrav=rand dim=128x128 repeat=1 reload=1 delay=1 outkind=sdl
show3: enc.ctx gen1.ctx gen2.ctx gen3.ctx
	tewel synth enc.ctx gen1.ctx gen2.ctx gen3.ctx src=/spin/dan/lq.128 \
		srctrav=rand dim=128x128 repeat=1 reload=1 delay=1 outkind=sdl
shownova: gen1.ctx gen2.ctx gen3.ctx nova2.rvg
	tewel synth gen1.ctx gen2.ctx gen3.ctx src=nova2.rvg \
		evolve=0.95 dim=1x1x1536 repeat=1 reload=1 delay=0.05 outkind=sdl
showpane: gen1.ctx gen2.ctx gen3.ctx shampane.rvg
	tewel synth gen1.ctx gen2.ctx gen3.ctx src=shampane.rvg \
		evolve=0.95 dim=1x1x1536 repeat=1 reload=1 delay=0.05 outkind=sdl




clean1:
	rm -f gen1.ctx enc.ctx src1.rgb tgt1.rgb learnfunc1.done
clean2:
	rm -f gen2.ctx dis2.ctx src2.rgb tgt2.rgb pgen2.ctx src2.tmp.rgb tgt2.tmp.rgb learnfunc2.done learnhans2.done
clean3:
	rm -f gen3.ctx dis3.ctx src3.rgb tgt3.rgb pgen3.ctx learnfunc3.done learnhans3.done

clean: clean1 clean2 clean3
	rm -f *.rvg *.ctx *.out
