??#
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements#
handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??"
?
embedding_9/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?' *'
shared_nameembedding_9/embeddings
?
*embedding_9/embeddings/Read/ReadVariableOpReadVariableOpembedding_9/embeddings*
_output_shapes
:	?' *
dtype0
z
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_18/kernel
s
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel*
_output_shapes

:
*
dtype0
r
dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_18/bias
k
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
_output_shapes
:
*
dtype0
z
dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_19/kernel
s
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel*
_output_shapes

:
*
dtype0
r
dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_19/bias
k
!dense_19/bias/Read/ReadVariableOpReadVariableOpdense_19/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
lstm_1/lstm_cell_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @**
shared_namelstm_1/lstm_cell_1/kernel
?
-lstm_1/lstm_cell_1/kernel/Read/ReadVariableOpReadVariableOplstm_1/lstm_cell_1/kernel*
_output_shapes

: @*
dtype0
?
#lstm_1/lstm_cell_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*4
shared_name%#lstm_1/lstm_cell_1/recurrent_kernel
?
7lstm_1/lstm_cell_1/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_1/lstm_cell_1/recurrent_kernel*
_output_shapes

:@*
dtype0
?
lstm_1/lstm_cell_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_namelstm_1/lstm_cell_1/bias

+lstm_1/lstm_cell_1/bias/Read/ReadVariableOpReadVariableOplstm_1/lstm_cell_1/bias*
_output_shapes
:@*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/embedding_9/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?' *.
shared_nameAdam/embedding_9/embeddings/m
?
1Adam/embedding_9/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_9/embeddings/m*
_output_shapes
:	?' *
dtype0
?
Adam/dense_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/dense_18/kernel/m
?
*Adam/dense_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/m*
_output_shapes

:
*
dtype0
?
Adam/dense_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_18/bias/m
y
(Adam/dense_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/bias/m*
_output_shapes
:
*
dtype0
?
Adam/dense_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/dense_19/kernel/m
?
*Adam/dense_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/m*
_output_shapes

:
*
dtype0
?
Adam/dense_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_19/bias/m
y
(Adam/dense_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_19/bias/m*
_output_shapes
:*
dtype0
?
 Adam/lstm_1/lstm_cell_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*1
shared_name" Adam/lstm_1/lstm_cell_1/kernel/m
?
4Adam/lstm_1/lstm_cell_1/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_1/lstm_cell_1/kernel/m*
_output_shapes

: @*
dtype0
?
*Adam/lstm_1/lstm_cell_1/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*;
shared_name,*Adam/lstm_1/lstm_cell_1/recurrent_kernel/m
?
>Adam/lstm_1/lstm_cell_1/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_1/lstm_cell_1/recurrent_kernel/m*
_output_shapes

:@*
dtype0
?
Adam/lstm_1/lstm_cell_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/lstm_1/lstm_cell_1/bias/m
?
2Adam/lstm_1/lstm_cell_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_1/lstm_cell_1/bias/m*
_output_shapes
:@*
dtype0
?
Adam/embedding_9/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?' *.
shared_nameAdam/embedding_9/embeddings/v
?
1Adam/embedding_9/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_9/embeddings/v*
_output_shapes
:	?' *
dtype0
?
Adam/dense_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/dense_18/kernel/v
?
*Adam/dense_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/v*
_output_shapes

:
*
dtype0
?
Adam/dense_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_18/bias/v
y
(Adam/dense_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/bias/v*
_output_shapes
:
*
dtype0
?
Adam/dense_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/dense_19/kernel/v
?
*Adam/dense_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/v*
_output_shapes

:
*
dtype0
?
Adam/dense_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_19/bias/v
y
(Adam/dense_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_19/bias/v*
_output_shapes
:*
dtype0
?
 Adam/lstm_1/lstm_cell_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*1
shared_name" Adam/lstm_1/lstm_cell_1/kernel/v
?
4Adam/lstm_1/lstm_cell_1/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_1/lstm_cell_1/kernel/v*
_output_shapes

: @*
dtype0
?
*Adam/lstm_1/lstm_cell_1/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*;
shared_name,*Adam/lstm_1/lstm_cell_1/recurrent_kernel/v
?
>Adam/lstm_1/lstm_cell_1/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_1/lstm_cell_1/recurrent_kernel/v*
_output_shapes

:@*
dtype0
?
Adam/lstm_1/lstm_cell_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/lstm_1/lstm_cell_1/bias/v
?
2Adam/lstm_1/lstm_cell_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_1/lstm_cell_1/bias/v*
_output_shapes
:@*
dtype0

NoOpNoOp
?3
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?3
value?3B?3 B?3
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer_with_weights-3
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
b

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
l
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
 	keras_api
h

!kernel
"bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
?
'iter

(beta_1

)beta_2
	*decay
+learning_ratemcmdme!mf"mg,mh-mi.mjvkvlvm!vn"vo,vp-vq.vr
8
0
,1
-2
.3
4
5
!6
"7
8
0
,1
-2
.3
4
5
!6
"7
 
?
/non_trainable_variables

0layers
1metrics
2layer_regularization_losses
3layer_metrics
	variables
trainable_variables
	regularization_losses
 
fd
VARIABLE_VALUEembedding_9/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
?
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses
?
9
state_size

,kernel
-recurrent_kernel
.bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
 

,0
-1
.2

,0
-1
.2
 
?

>states
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEdense_18/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_18/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
?
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEdense_19/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_19/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1

!0
"1
 
?
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
#	variables
$trainable_variables
%regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_1/lstm_cell_1/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#lstm_1/lstm_cell_1/recurrent_kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUElstm_1/lstm_cell_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
 
#
0
1
2
3
4

S0
T1
 
 
 
 
 
 
 
 

,0
-1
.2

,0
-1
.2
 
?
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
:	variables
;trainable_variables
<regularization_losses
 
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Ztotal
	[count
\	variables
]	keras_api
D
	^total
	_count
`
_fn_kwargs
a	variables
b	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Z0
[1

\	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

^0
_1

a	variables
??
VARIABLE_VALUEAdam/embedding_9/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_18/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_18/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_19/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_19/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_1/lstm_cell_1/kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_1/lstm_cell_1/recurrent_kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_1/lstm_cell_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/embedding_9/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_18/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_18/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_19/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_19/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_1/lstm_cell_1/kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_1/lstm_cell_1/recurrent_kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_1/lstm_cell_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
!serving_default_embedding_9_inputPlaceholder*0
_output_shapes
:??????????????????*
dtype0*%
shape:??????????????????
?
StatefulPartitionedCallStatefulPartitionedCall!serving_default_embedding_9_inputembedding_9/embeddingslstm_1/lstm_cell_1/kernellstm_1/lstm_cell_1/bias#lstm_1/lstm_cell_1/recurrent_kerneldense_18/kerneldense_18/biasdense_19/kerneldense_19/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_162199
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*embedding_9/embeddings/Read/ReadVariableOp#dense_18/kernel/Read/ReadVariableOp!dense_18/bias/Read/ReadVariableOp#dense_19/kernel/Read/ReadVariableOp!dense_19/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp-lstm_1/lstm_cell_1/kernel/Read/ReadVariableOp7lstm_1/lstm_cell_1/recurrent_kernel/Read/ReadVariableOp+lstm_1/lstm_cell_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp1Adam/embedding_9/embeddings/m/Read/ReadVariableOp*Adam/dense_18/kernel/m/Read/ReadVariableOp(Adam/dense_18/bias/m/Read/ReadVariableOp*Adam/dense_19/kernel/m/Read/ReadVariableOp(Adam/dense_19/bias/m/Read/ReadVariableOp4Adam/lstm_1/lstm_cell_1/kernel/m/Read/ReadVariableOp>Adam/lstm_1/lstm_cell_1/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_1/lstm_cell_1/bias/m/Read/ReadVariableOp1Adam/embedding_9/embeddings/v/Read/ReadVariableOp*Adam/dense_18/kernel/v/Read/ReadVariableOp(Adam/dense_18/bias/v/Read/ReadVariableOp*Adam/dense_19/kernel/v/Read/ReadVariableOp(Adam/dense_19/bias/v/Read/ReadVariableOp4Adam/lstm_1/lstm_cell_1/kernel/v/Read/ReadVariableOp>Adam/lstm_1/lstm_cell_1/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_1/lstm_cell_1/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_164644
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_9/embeddingsdense_18/kerneldense_18/biasdense_19/kerneldense_19/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_1/lstm_cell_1/kernel#lstm_1/lstm_cell_1/recurrent_kernellstm_1/lstm_cell_1/biastotalcounttotal_1count_1Adam/embedding_9/embeddings/mAdam/dense_18/kernel/mAdam/dense_18/bias/mAdam/dense_19/kernel/mAdam/dense_19/bias/m Adam/lstm_1/lstm_cell_1/kernel/m*Adam/lstm_1/lstm_cell_1/recurrent_kernel/mAdam/lstm_1/lstm_cell_1/bias/mAdam/embedding_9/embeddings/vAdam/dense_18/kernel/vAdam/dense_18/bias/vAdam/dense_19/kernel/vAdam/dense_19/bias/v Adam/lstm_1/lstm_cell_1/kernel/v*Adam/lstm_1/lstm_cell_1/recurrent_kernel/vAdam/lstm_1/lstm_cell_1/bias/v*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_164753??!
?
?
'__inference_lstm_1_layer_call_fn_162965

inputs
unknown: @
	unknown_0:@
	unknown_1:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_162015o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????????????? : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?D
?
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_164376

inputs
states_0
states_1/
split_readvariableop_resource: @-
split_1_readvariableop_resource:@)
readvariableop_resource:@
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:????????? I
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??}
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????X
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:????????? Z
mul_1Mulinputsones_like:output:0*
T0*'
_output_shapes
:????????? Z
mul_2Mulinputsones_like:output:0*
T0*'
_output_shapes
:????????? Z
mul_3Mulinputsones_like:output:0*
T0*'
_output_shapes
:????????? Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :r
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

: @*
dtype0?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split[
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:?????????_
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:?????????_
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:?????????_
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:?????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:@*
dtype0?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????^
mul_4Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:?????????^
mul_5Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:?????????^
mul_6Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:?????????^
mul_7Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:?????????f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskg
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:?????????d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:?????????h
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????W
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????h
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????U
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????V
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:?????????h
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????h
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????Q
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:?????????K
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:?????????Z
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????Y
IdentityIdentity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:?????????[

Identity_1Identity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:?????????Z

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:????????? :?????????:?????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/1
?	
?
$__inference_signature_wrapper_162199
embedding_9_input
unknown:	?' 
	unknown_0: @
	unknown_1:@
	unknown_2:@
	unknown_3:

	unknown_4:

	unknown_5:

	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallembedding_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_160742o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
0
_output_shapes
:??????????????????
+
_user_specified_nameembedding_9_input
Ȁ
?
B__inference_lstm_1_layer_call_and_return_conditional_losses_163208
inputs_0;
)lstm_cell_1_split_readvariableop_resource: @9
+lstm_cell_1_split_1_readvariableop_resource:@5
#lstm_cell_1_readvariableop_resource:@
identity??lstm_cell_1/ReadVariableOp?lstm_cell_1/ReadVariableOp_1?lstm_cell_1/ReadVariableOp_2?lstm_cell_1/ReadVariableOp_3? lstm_cell_1/split/ReadVariableOp?"lstm_cell_1/split_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_maskc
lstm_cell_1/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:`
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? [
lstm_cell_1/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:b
lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/ones_like_1Fill&lstm_cell_1/ones_like_1/Shape:output:0&lstm_cell_1/ones_like_1/Const:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/mulMulstrided_slice_2:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/mul_1Mulstrided_slice_2:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/mul_2Mulstrided_slice_2:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/mul_3Mulstrided_slice_2:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_1/split/ReadVariableOpReadVariableOp)lstm_cell_1_split_readvariableop_resource*
_output_shapes

: @*
dtype0?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0(lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split
lstm_cell_1/MatMulMatMullstm_cell_1/mul:z:0lstm_cell_1/split:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/MatMul_1MatMullstm_cell_1/mul_1:z:0lstm_cell_1/split:output:1*
T0*'
_output_shapes
:??????????
lstm_cell_1/MatMul_2MatMullstm_cell_1/mul_2:z:0lstm_cell_1/split:output:2*
T0*'
_output_shapes
:??????????
lstm_cell_1/MatMul_3MatMullstm_cell_1/mul_3:z:0lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????_
lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_1/split_1/ReadVariableOpReadVariableOp+lstm_cell_1_split_1_readvariableop_resource*
_output_shapes
:@*
dtype0?
lstm_cell_1/split_1Split&lstm_cell_1/split_1/split_dim:output:0*lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/MatMul:product:0lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/BiasAdd_1BiasAddlstm_cell_1/MatMul_1:product:0lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:??????????
lstm_cell_1/BiasAdd_2BiasAddlstm_cell_1/MatMul_2:product:0lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
lstm_cell_1/BiasAdd_3BiasAddlstm_cell_1/MatMul_3:product:0lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????|
lstm_cell_1/mul_4Mulzeros:output:0 lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:?????????|
lstm_cell_1/mul_5Mulzeros:output:0 lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:?????????|
lstm_cell_1/mul_6Mulzeros:output:0 lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:?????????|
lstm_cell_1/mul_7Mulzeros:output:0 lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:?????????~
lstm_cell_1/ReadVariableOpReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0p
lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_1/strided_sliceStridedSlice"lstm_cell_1/ReadVariableOp:value:0(lstm_cell_1/strided_slice/stack:output:0*lstm_cell_1/strided_slice/stack_1:output:0*lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_1/MatMul_4MatMullstm_cell_1/mul_4:z:0"lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/addAddV2lstm_cell_1/BiasAdd:output:0lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????e
lstm_cell_1/SigmoidSigmoidlstm_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/ReadVariableOp_1ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0r
!lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_1/strided_slice_1StridedSlice$lstm_cell_1/ReadVariableOp_1:value:0*lstm_cell_1/strided_slice_1/stack:output:0,lstm_cell_1/strided_slice_1/stack_1:output:0,lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_1/MatMul_5MatMullstm_cell_1/mul_5:z:0$lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/add_1AddV2lstm_cell_1/BiasAdd_1:output:0lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????i
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????w
lstm_cell_1/mul_8Mullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/ReadVariableOp_2ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0r
!lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   t
#lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_1/strided_slice_2StridedSlice$lstm_cell_1/ReadVariableOp_2:value:0*lstm_cell_1/strided_slice_2/stack:output:0,lstm_cell_1/strided_slice_2/stack_1:output:0,lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_1/MatMul_6MatMullstm_cell_1/mul_6:z:0$lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/add_2AddV2lstm_cell_1/BiasAdd_2:output:0lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????a
lstm_cell_1/TanhTanhlstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????y
lstm_cell_1/mul_9Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????z
lstm_cell_1/add_3AddV2lstm_cell_1/mul_8:z:0lstm_cell_1/mul_9:z:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/ReadVariableOp_3ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0r
!lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   t
#lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_1/strided_slice_3StridedSlice$lstm_cell_1/ReadVariableOp_3:value:0*lstm_cell_1/strided_slice_3/stack:output:0,lstm_cell_1/strided_slice_3/stack_1:output:0,lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_1/MatMul_7MatMullstm_cell_1/mul_7:z:0$lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/add_4AddV2lstm_cell_1/BiasAdd_3:output:0lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????i
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????c
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????~
lstm_cell_1/mul_10Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_1_split_readvariableop_resource+lstm_cell_1_split_1_readvariableop_resource#lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_163074*
condR
while_cond_163073*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????????????? : : : 28
lstm_cell_1/ReadVariableOplstm_cell_1/ReadVariableOp2<
lstm_cell_1/ReadVariableOp_1lstm_cell_1/ReadVariableOp_12<
lstm_cell_1/ReadVariableOp_2lstm_cell_1/ReadVariableOp_22<
lstm_cell_1/ReadVariableOp_3lstm_cell_1/ReadVariableOp_32D
 lstm_cell_1/split/ReadVariableOp lstm_cell_1/split/ReadVariableOp2H
"lstm_cell_1/split_1/ReadVariableOp"lstm_cell_1/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :?????????????????? 
"
_user_specified_name
inputs/0
?

?
D__inference_dense_18_layer_call_and_return_conditional_losses_164213

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_lstm_cell_1_layer_call_fn_164277

inputs
states_0
states_1
unknown: @
	unknown_0:@
	unknown_1:@
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_160859o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:????????? :?????????:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/1
?G
?
__inference__traced_save_164644
file_prefix5
1savev2_embedding_9_embeddings_read_readvariableop.
*savev2_dense_18_kernel_read_readvariableop,
(savev2_dense_18_bias_read_readvariableop.
*savev2_dense_19_kernel_read_readvariableop,
(savev2_dense_19_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop8
4savev2_lstm_1_lstm_cell_1_kernel_read_readvariableopB
>savev2_lstm_1_lstm_cell_1_recurrent_kernel_read_readvariableop6
2savev2_lstm_1_lstm_cell_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop<
8savev2_adam_embedding_9_embeddings_m_read_readvariableop5
1savev2_adam_dense_18_kernel_m_read_readvariableop3
/savev2_adam_dense_18_bias_m_read_readvariableop5
1savev2_adam_dense_19_kernel_m_read_readvariableop3
/savev2_adam_dense_19_bias_m_read_readvariableop?
;savev2_adam_lstm_1_lstm_cell_1_kernel_m_read_readvariableopI
Esavev2_adam_lstm_1_lstm_cell_1_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_1_lstm_cell_1_bias_m_read_readvariableop<
8savev2_adam_embedding_9_embeddings_v_read_readvariableop5
1savev2_adam_dense_18_kernel_v_read_readvariableop3
/savev2_adam_dense_18_bias_v_read_readvariableop5
1savev2_adam_dense_19_kernel_v_read_readvariableop3
/savev2_adam_dense_19_bias_v_read_readvariableop?
;savev2_adam_lstm_1_lstm_cell_1_kernel_v_read_readvariableopI
Esavev2_adam_lstm_1_lstm_cell_1_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_1_lstm_cell_1_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_embedding_9_embeddings_read_readvariableop*savev2_dense_18_kernel_read_readvariableop(savev2_dense_18_bias_read_readvariableop*savev2_dense_19_kernel_read_readvariableop(savev2_dense_19_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop4savev2_lstm_1_lstm_cell_1_kernel_read_readvariableop>savev2_lstm_1_lstm_cell_1_recurrent_kernel_read_readvariableop2savev2_lstm_1_lstm_cell_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop8savev2_adam_embedding_9_embeddings_m_read_readvariableop1savev2_adam_dense_18_kernel_m_read_readvariableop/savev2_adam_dense_18_bias_m_read_readvariableop1savev2_adam_dense_19_kernel_m_read_readvariableop/savev2_adam_dense_19_bias_m_read_readvariableop;savev2_adam_lstm_1_lstm_cell_1_kernel_m_read_readvariableopEsavev2_adam_lstm_1_lstm_cell_1_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_1_lstm_cell_1_bias_m_read_readvariableop8savev2_adam_embedding_9_embeddings_v_read_readvariableop1savev2_adam_dense_18_kernel_v_read_readvariableop/savev2_adam_dense_18_bias_v_read_readvariableop1savev2_adam_dense_19_kernel_v_read_readvariableop/savev2_adam_dense_19_bias_v_read_readvariableop;savev2_adam_lstm_1_lstm_cell_1_kernel_v_read_readvariableopEsavev2_adam_lstm_1_lstm_cell_1_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_1_lstm_cell_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?' :
:
:
:: : : : : : @:@:@: : : : :	?' :
:
:
:: @:@:@:	?' :
:
:
:: @:@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?' :$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :$ 

_output_shapes

: @:$ 

_output_shapes

:@: 

_output_shapes
:@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?' :$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

: @:$ 

_output_shapes

:@: 

_output_shapes
:@:%!

_output_shapes
:	?' :$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

: @:$  

_output_shapes

:@: !

_output_shapes
:@:"

_output_shapes
: 
??
?
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_164522

inputs
states_0
states_1/
split_readvariableop_resource: @-
split_1_readvariableop_resource:@)
readvariableop_resource:@
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:????????? R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??p
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:????????? O
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? T
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??t
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:????????? Q
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? s
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? o
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:????????? T
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??t
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:????????? Q
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? s
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? o
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:????????? T
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??t
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:????????? Q
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? s
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? o
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:????????? I
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??}
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????T
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??v
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*'
_output_shapes
:?????????S
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:?
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???]
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????s
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????o
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????T
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??v
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*'
_output_shapes
:?????????S
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:?
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2?ǔ]
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????s
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????o
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????T
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??v
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*'
_output_shapes
:?????????S
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:?
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2??U]
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????s
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????o
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*'
_output_shapes
:?????????T
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??v
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*'
_output_shapes
:?????????S
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:?
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2ꪬ]
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????s
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????o
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*'
_output_shapes
:?????????W
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? [
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:????????? [
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*'
_output_shapes
:????????? [
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:????????? Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :r
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

: @*
dtype0?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split[
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:?????????_
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:?????????_
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:?????????_
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:?????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:@*
dtype0?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????]
mul_4Mulstates_0dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:?????????]
mul_5Mulstates_0dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:?????????]
mul_6Mulstates_0dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:?????????]
mul_7Mulstates_0dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:?????????f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskg
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:?????????d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:?????????h
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????W
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????h
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????U
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????V
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:?????????h
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????h
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????Q
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:?????????K
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:?????????Z
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????Y
IdentityIdentity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:?????????[

Identity_1Identity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:?????????Z

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:????????? :?????????:?????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/1
?
?
while_cond_163687
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_163687___redundant_placeholder04
0while_while_cond_163687___redundant_placeholder14
0while_while_cond_163687___redundant_placeholder24
0while_while_cond_163687___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
'__inference_lstm_1_layer_call_fn_162932
inputs_0
unknown: @
	unknown_0:@
	unknown_1:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_160942o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????????????? : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :?????????????????? 
"
_user_specified_name
inputs/0
?
?
,__inference_lstm_cell_1_layer_call_fn_164294

inputs
states_0
states_1
unknown: @
	unknown_0:@
	unknown_1:@
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_161119o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:????????? :?????????:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/1
??
?
lstm_1_while_body_162684*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3)
%lstm_1_while_lstm_1_strided_slice_1_0e
alstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0J
8lstm_1_while_lstm_cell_1_split_readvariableop_resource_0: @H
:lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_0:@D
2lstm_1_while_lstm_cell_1_readvariableop_resource_0:@
lstm_1_while_identity
lstm_1_while_identity_1
lstm_1_while_identity_2
lstm_1_while_identity_3
lstm_1_while_identity_4
lstm_1_while_identity_5'
#lstm_1_while_lstm_1_strided_slice_1c
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensorH
6lstm_1_while_lstm_cell_1_split_readvariableop_resource: @F
8lstm_1_while_lstm_cell_1_split_1_readvariableop_resource:@B
0lstm_1_while_lstm_cell_1_readvariableop_resource:@??'lstm_1/while/lstm_cell_1/ReadVariableOp?)lstm_1/while/lstm_cell_1/ReadVariableOp_1?)lstm_1/while/lstm_cell_1/ReadVariableOp_2?)lstm_1/while/lstm_cell_1/ReadVariableOp_3?-lstm_1/while/lstm_cell_1/split/ReadVariableOp?/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp?
>lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
0lstm_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0lstm_1_while_placeholderGlstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype0?
(lstm_1/while/lstm_cell_1/ones_like/ShapeShape7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:m
(lstm_1/while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
"lstm_1/while/lstm_cell_1/ones_likeFill1lstm_1/while/lstm_cell_1/ones_like/Shape:output:01lstm_1/while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? k
&lstm_1/while/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
$lstm_1/while/lstm_cell_1/dropout/MulMul+lstm_1/while/lstm_cell_1/ones_like:output:0/lstm_1/while/lstm_cell_1/dropout/Const:output:0*
T0*'
_output_shapes
:????????? ?
&lstm_1/while/lstm_cell_1/dropout/ShapeShape+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
=lstm_1/while/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform/lstm_1/while/lstm_cell_1/dropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???t
/lstm_1/while/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
-lstm_1/while/lstm_cell_1/dropout/GreaterEqualGreaterEqualFlstm_1/while/lstm_cell_1/dropout/random_uniform/RandomUniform:output:08lstm_1/while/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
%lstm_1/while/lstm_cell_1/dropout/CastCast1lstm_1/while/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
&lstm_1/while/lstm_cell_1/dropout/Mul_1Mul(lstm_1/while/lstm_cell_1/dropout/Mul:z:0)lstm_1/while/lstm_cell_1/dropout/Cast:y:0*
T0*'
_output_shapes
:????????? m
(lstm_1/while/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
&lstm_1/while/lstm_cell_1/dropout_1/MulMul+lstm_1/while/lstm_cell_1/ones_like:output:01lstm_1/while/lstm_cell_1/dropout_1/Const:output:0*
T0*'
_output_shapes
:????????? ?
(lstm_1/while/lstm_cell_1/dropout_1/ShapeShape+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
?lstm_1/while/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform1lstm_1/while/lstm_cell_1/dropout_1/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2?їv
1lstm_1/while/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
/lstm_1/while/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualHlstm_1/while/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:0:lstm_1/while/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
'lstm_1/while/lstm_cell_1/dropout_1/CastCast3lstm_1/while/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
(lstm_1/while/lstm_cell_1/dropout_1/Mul_1Mul*lstm_1/while/lstm_cell_1/dropout_1/Mul:z:0+lstm_1/while/lstm_cell_1/dropout_1/Cast:y:0*
T0*'
_output_shapes
:????????? m
(lstm_1/while/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
&lstm_1/while/lstm_cell_1/dropout_2/MulMul+lstm_1/while/lstm_cell_1/ones_like:output:01lstm_1/while/lstm_cell_1/dropout_2/Const:output:0*
T0*'
_output_shapes
:????????? ?
(lstm_1/while/lstm_cell_1/dropout_2/ShapeShape+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
?lstm_1/while/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform1lstm_1/while/lstm_cell_1/dropout_2/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???v
1lstm_1/while/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
/lstm_1/while/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualHlstm_1/while/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:0:lstm_1/while/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
'lstm_1/while/lstm_cell_1/dropout_2/CastCast3lstm_1/while/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
(lstm_1/while/lstm_cell_1/dropout_2/Mul_1Mul*lstm_1/while/lstm_cell_1/dropout_2/Mul:z:0+lstm_1/while/lstm_cell_1/dropout_2/Cast:y:0*
T0*'
_output_shapes
:????????? m
(lstm_1/while/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
&lstm_1/while/lstm_cell_1/dropout_3/MulMul+lstm_1/while/lstm_cell_1/ones_like:output:01lstm_1/while/lstm_cell_1/dropout_3/Const:output:0*
T0*'
_output_shapes
:????????? ?
(lstm_1/while/lstm_cell_1/dropout_3/ShapeShape+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
?lstm_1/while/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform1lstm_1/while/lstm_cell_1/dropout_3/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2?ܥv
1lstm_1/while/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
/lstm_1/while/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualHlstm_1/while/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:0:lstm_1/while/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
'lstm_1/while/lstm_cell_1/dropout_3/CastCast3lstm_1/while/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
(lstm_1/while/lstm_cell_1/dropout_3/Mul_1Mul*lstm_1/while/lstm_cell_1/dropout_3/Mul:z:0+lstm_1/while/lstm_cell_1/dropout_3/Cast:y:0*
T0*'
_output_shapes
:????????? t
*lstm_1/while/lstm_cell_1/ones_like_1/ShapeShapelstm_1_while_placeholder_2*
T0*
_output_shapes
:o
*lstm_1/while/lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
$lstm_1/while/lstm_cell_1/ones_like_1Fill3lstm_1/while/lstm_cell_1/ones_like_1/Shape:output:03lstm_1/while/lstm_cell_1/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????m
(lstm_1/while/lstm_cell_1/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
&lstm_1/while/lstm_cell_1/dropout_4/MulMul-lstm_1/while/lstm_cell_1/ones_like_1:output:01lstm_1/while/lstm_cell_1/dropout_4/Const:output:0*
T0*'
_output_shapes
:??????????
(lstm_1/while/lstm_cell_1/dropout_4/ShapeShape-lstm_1/while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
?lstm_1/while/lstm_cell_1/dropout_4/random_uniform/RandomUniformRandomUniform1lstm_1/while/lstm_cell_1/dropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2??v
1lstm_1/while/lstm_cell_1/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
/lstm_1/while/lstm_cell_1/dropout_4/GreaterEqualGreaterEqualHlstm_1/while/lstm_cell_1/dropout_4/random_uniform/RandomUniform:output:0:lstm_1/while/lstm_cell_1/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
'lstm_1/while/lstm_cell_1/dropout_4/CastCast3lstm_1/while/lstm_cell_1/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
(lstm_1/while/lstm_cell_1/dropout_4/Mul_1Mul*lstm_1/while/lstm_cell_1/dropout_4/Mul:z:0+lstm_1/while/lstm_cell_1/dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????m
(lstm_1/while/lstm_cell_1/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
&lstm_1/while/lstm_cell_1/dropout_5/MulMul-lstm_1/while/lstm_cell_1/ones_like_1:output:01lstm_1/while/lstm_cell_1/dropout_5/Const:output:0*
T0*'
_output_shapes
:??????????
(lstm_1/while/lstm_cell_1/dropout_5/ShapeShape-lstm_1/while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
?lstm_1/while/lstm_cell_1/dropout_5/random_uniform/RandomUniformRandomUniform1lstm_1/while/lstm_cell_1/dropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???v
1lstm_1/while/lstm_cell_1/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
/lstm_1/while/lstm_cell_1/dropout_5/GreaterEqualGreaterEqualHlstm_1/while/lstm_cell_1/dropout_5/random_uniform/RandomUniform:output:0:lstm_1/while/lstm_cell_1/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
'lstm_1/while/lstm_cell_1/dropout_5/CastCast3lstm_1/while/lstm_cell_1/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
(lstm_1/while/lstm_cell_1/dropout_5/Mul_1Mul*lstm_1/while/lstm_cell_1/dropout_5/Mul:z:0+lstm_1/while/lstm_cell_1/dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????m
(lstm_1/while/lstm_cell_1/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
&lstm_1/while/lstm_cell_1/dropout_6/MulMul-lstm_1/while/lstm_cell_1/ones_like_1:output:01lstm_1/while/lstm_cell_1/dropout_6/Const:output:0*
T0*'
_output_shapes
:??????????
(lstm_1/while/lstm_cell_1/dropout_6/ShapeShape-lstm_1/while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
?lstm_1/while/lstm_cell_1/dropout_6/random_uniform/RandomUniformRandomUniform1lstm_1/while/lstm_cell_1/dropout_6/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2??v
1lstm_1/while/lstm_cell_1/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
/lstm_1/while/lstm_cell_1/dropout_6/GreaterEqualGreaterEqualHlstm_1/while/lstm_cell_1/dropout_6/random_uniform/RandomUniform:output:0:lstm_1/while/lstm_cell_1/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
'lstm_1/while/lstm_cell_1/dropout_6/CastCast3lstm_1/while/lstm_cell_1/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
(lstm_1/while/lstm_cell_1/dropout_6/Mul_1Mul*lstm_1/while/lstm_cell_1/dropout_6/Mul:z:0+lstm_1/while/lstm_cell_1/dropout_6/Cast:y:0*
T0*'
_output_shapes
:?????????m
(lstm_1/while/lstm_cell_1/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
&lstm_1/while/lstm_cell_1/dropout_7/MulMul-lstm_1/while/lstm_cell_1/ones_like_1:output:01lstm_1/while/lstm_cell_1/dropout_7/Const:output:0*
T0*'
_output_shapes
:??????????
(lstm_1/while/lstm_cell_1/dropout_7/ShapeShape-lstm_1/while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
?lstm_1/while/lstm_cell_1/dropout_7/random_uniform/RandomUniformRandomUniform1lstm_1/while/lstm_cell_1/dropout_7/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???v
1lstm_1/while/lstm_cell_1/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
/lstm_1/while/lstm_cell_1/dropout_7/GreaterEqualGreaterEqualHlstm_1/while/lstm_cell_1/dropout_7/random_uniform/RandomUniform:output:0:lstm_1/while/lstm_cell_1/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
'lstm_1/while/lstm_cell_1/dropout_7/CastCast3lstm_1/while/lstm_cell_1/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
(lstm_1/while/lstm_cell_1/dropout_7/Mul_1Mul*lstm_1/while/lstm_cell_1/dropout_7/Mul:z:0+lstm_1/while/lstm_cell_1/dropout_7/Cast:y:0*
T0*'
_output_shapes
:??????????
lstm_1/while/lstm_cell_1/mulMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0*lstm_1/while/lstm_cell_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
lstm_1/while/lstm_cell_1/mul_1Mul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0,lstm_1/while/lstm_cell_1/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
lstm_1/while/lstm_cell_1/mul_2Mul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0,lstm_1/while/lstm_cell_1/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
lstm_1/while/lstm_cell_1/mul_3Mul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0,lstm_1/while/lstm_cell_1/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:????????? j
(lstm_1/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
-lstm_1/while/lstm_cell_1/split/ReadVariableOpReadVariableOp8lstm_1_while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes

: @*
dtype0?
lstm_1/while/lstm_cell_1/splitSplit1lstm_1/while/lstm_cell_1/split/split_dim:output:05lstm_1/while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split?
lstm_1/while/lstm_cell_1/MatMulMatMul lstm_1/while/lstm_cell_1/mul:z:0'lstm_1/while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:??????????
!lstm_1/while/lstm_cell_1/MatMul_1MatMul"lstm_1/while/lstm_cell_1/mul_1:z:0'lstm_1/while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:??????????
!lstm_1/while/lstm_cell_1/MatMul_2MatMul"lstm_1/while/lstm_cell_1/mul_2:z:0'lstm_1/while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:??????????
!lstm_1/while/lstm_cell_1/MatMul_3MatMul"lstm_1/while/lstm_cell_1/mul_3:z:0'lstm_1/while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????l
*lstm_1/while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
/lstm_1/while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp:lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
 lstm_1/while/lstm_cell_1/split_1Split3lstm_1/while/lstm_cell_1/split_1/split_dim:output:07lstm_1/while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
 lstm_1/while/lstm_cell_1/BiasAddBiasAdd)lstm_1/while/lstm_cell_1/MatMul:product:0)lstm_1/while/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:??????????
"lstm_1/while/lstm_cell_1/BiasAdd_1BiasAdd+lstm_1/while/lstm_cell_1/MatMul_1:product:0)lstm_1/while/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:??????????
"lstm_1/while/lstm_cell_1/BiasAdd_2BiasAdd+lstm_1/while/lstm_cell_1/MatMul_2:product:0)lstm_1/while/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
"lstm_1/while/lstm_cell_1/BiasAdd_3BiasAdd+lstm_1/while/lstm_cell_1/MatMul_3:product:0)lstm_1/while/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:??????????
lstm_1/while/lstm_cell_1/mul_4Mullstm_1_while_placeholder_2,lstm_1/while/lstm_cell_1/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:??????????
lstm_1/while/lstm_cell_1/mul_5Mullstm_1_while_placeholder_2,lstm_1/while/lstm_cell_1/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:??????????
lstm_1/while/lstm_cell_1/mul_6Mullstm_1_while_placeholder_2,lstm_1/while/lstm_cell_1/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:??????????
lstm_1/while/lstm_cell_1/mul_7Mullstm_1_while_placeholder_2,lstm_1/while/lstm_cell_1/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:??????????
'lstm_1/while/lstm_cell_1/ReadVariableOpReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0}
,lstm_1/while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.lstm_1/while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.lstm_1/while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
&lstm_1/while/lstm_cell_1/strided_sliceStridedSlice/lstm_1/while/lstm_cell_1/ReadVariableOp:value:05lstm_1/while/lstm_cell_1/strided_slice/stack:output:07lstm_1/while/lstm_cell_1/strided_slice/stack_1:output:07lstm_1/while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
!lstm_1/while/lstm_cell_1/MatMul_4MatMul"lstm_1/while/lstm_cell_1/mul_4:z:0/lstm_1/while/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:??????????
lstm_1/while/lstm_cell_1/addAddV2)lstm_1/while/lstm_cell_1/BiasAdd:output:0+lstm_1/while/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????
 lstm_1/while/lstm_cell_1/SigmoidSigmoid lstm_1/while/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
)lstm_1/while/lstm_cell_1/ReadVariableOp_1ReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0
.lstm_1/while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       ?
0lstm_1/while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
0lstm_1/while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_1/while/lstm_cell_1/strided_slice_1StridedSlice1lstm_1/while/lstm_cell_1/ReadVariableOp_1:value:07lstm_1/while/lstm_cell_1/strided_slice_1/stack:output:09lstm_1/while/lstm_cell_1/strided_slice_1/stack_1:output:09lstm_1/while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
!lstm_1/while/lstm_cell_1/MatMul_5MatMul"lstm_1/while/lstm_cell_1/mul_5:z:01lstm_1/while/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
lstm_1/while/lstm_cell_1/add_1AddV2+lstm_1/while/lstm_cell_1/BiasAdd_1:output:0+lstm_1/while/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:??????????
"lstm_1/while/lstm_cell_1/Sigmoid_1Sigmoid"lstm_1/while/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:??????????
lstm_1/while/lstm_cell_1/mul_8Mul&lstm_1/while/lstm_cell_1/Sigmoid_1:y:0lstm_1_while_placeholder_3*
T0*'
_output_shapes
:??????????
)lstm_1/while/lstm_cell_1/ReadVariableOp_2ReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0
.lstm_1/while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
0lstm_1/while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   ?
0lstm_1/while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_1/while/lstm_cell_1/strided_slice_2StridedSlice1lstm_1/while/lstm_cell_1/ReadVariableOp_2:value:07lstm_1/while/lstm_cell_1/strided_slice_2/stack:output:09lstm_1/while/lstm_cell_1/strided_slice_2/stack_1:output:09lstm_1/while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
!lstm_1/while/lstm_cell_1/MatMul_6MatMul"lstm_1/while/lstm_cell_1/mul_6:z:01lstm_1/while/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
lstm_1/while/lstm_cell_1/add_2AddV2+lstm_1/while/lstm_cell_1/BiasAdd_2:output:0+lstm_1/while/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????{
lstm_1/while/lstm_cell_1/TanhTanh"lstm_1/while/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:??????????
lstm_1/while/lstm_cell_1/mul_9Mul$lstm_1/while/lstm_cell_1/Sigmoid:y:0!lstm_1/while/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:??????????
lstm_1/while/lstm_cell_1/add_3AddV2"lstm_1/while/lstm_cell_1/mul_8:z:0"lstm_1/while/lstm_cell_1/mul_9:z:0*
T0*'
_output_shapes
:??????????
)lstm_1/while/lstm_cell_1/ReadVariableOp_3ReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0
.lstm_1/while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   ?
0lstm_1/while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
0lstm_1/while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_1/while/lstm_cell_1/strided_slice_3StridedSlice1lstm_1/while/lstm_cell_1/ReadVariableOp_3:value:07lstm_1/while/lstm_cell_1/strided_slice_3/stack:output:09lstm_1/while/lstm_cell_1/strided_slice_3/stack_1:output:09lstm_1/while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
!lstm_1/while/lstm_cell_1/MatMul_7MatMul"lstm_1/while/lstm_cell_1/mul_7:z:01lstm_1/while/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
lstm_1/while/lstm_cell_1/add_4AddV2+lstm_1/while/lstm_cell_1/BiasAdd_3:output:0+lstm_1/while/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:??????????
"lstm_1/while/lstm_cell_1/Sigmoid_2Sigmoid"lstm_1/while/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????}
lstm_1/while/lstm_cell_1/Tanh_1Tanh"lstm_1/while/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:??????????
lstm_1/while/lstm_cell_1/mul_10Mul&lstm_1/while/lstm_cell_1/Sigmoid_2:y:0#lstm_1/while/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:??????????
1lstm_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_1_while_placeholder_1lstm_1_while_placeholder#lstm_1/while/lstm_cell_1/mul_10:z:0*
_output_shapes
: *
element_dtype0:???T
lstm_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_1/while/addAddV2lstm_1_while_placeholderlstm_1/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_1/while/add_1AddV2&lstm_1_while_lstm_1_while_loop_counterlstm_1/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_1/while/IdentityIdentitylstm_1/while/add_1:z:0^lstm_1/while/NoOp*
T0*
_output_shapes
: ?
lstm_1/while/Identity_1Identity,lstm_1_while_lstm_1_while_maximum_iterations^lstm_1/while/NoOp*
T0*
_output_shapes
: n
lstm_1/while/Identity_2Identitylstm_1/while/add:z:0^lstm_1/while/NoOp*
T0*
_output_shapes
: ?
lstm_1/while/Identity_3IdentityAlstm_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_1/while/NoOp*
T0*
_output_shapes
: ?
lstm_1/while/Identity_4Identity#lstm_1/while/lstm_cell_1/mul_10:z:0^lstm_1/while/NoOp*
T0*'
_output_shapes
:??????????
lstm_1/while/Identity_5Identity"lstm_1/while/lstm_cell_1/add_3:z:0^lstm_1/while/NoOp*
T0*'
_output_shapes
:??????????
lstm_1/while/NoOpNoOp(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_1_while_identitylstm_1/while/Identity:output:0";
lstm_1_while_identity_1 lstm_1/while/Identity_1:output:0";
lstm_1_while_identity_2 lstm_1/while/Identity_2:output:0";
lstm_1_while_identity_3 lstm_1/while/Identity_3:output:0";
lstm_1_while_identity_4 lstm_1/while/Identity_4:output:0";
lstm_1_while_identity_5 lstm_1/while/Identity_5:output:0"L
#lstm_1_while_lstm_1_strided_slice_1%lstm_1_while_lstm_1_strided_slice_1_0"f
0lstm_1_while_lstm_cell_1_readvariableop_resource2lstm_1_while_lstm_cell_1_readvariableop_resource_0"v
8lstm_1_while_lstm_cell_1_split_1_readvariableop_resource:lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_0"r
6lstm_1_while_lstm_cell_1_split_readvariableop_resource8lstm_1_while_lstm_cell_1_split_readvariableop_resource_0"?
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensoralstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????:?????????: : : : : 2R
'lstm_1/while/lstm_cell_1/ReadVariableOp'lstm_1/while/lstm_cell_1/ReadVariableOp2V
)lstm_1/while/lstm_cell_1/ReadVariableOp_1)lstm_1/while/lstm_cell_1/ReadVariableOp_12V
)lstm_1/while/lstm_cell_1/ReadVariableOp_2)lstm_1/while/lstm_cell_1/ReadVariableOp_22V
)lstm_1/while/lstm_cell_1/ReadVariableOp_3)lstm_1/while/lstm_cell_1/ReadVariableOp_32^
-lstm_1/while/lstm_cell_1/split/ReadVariableOp-lstm_1/while/lstm_cell_1/split/ReadVariableOp2b
/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_162145
embedding_9_input%
embedding_9_162123:	?' 
lstm_1_162126: @
lstm_1_162128:@
lstm_1_162130:@!
dense_18_162133:

dense_18_162135:
!
dense_19_162139:

dense_19_162141:
identity?? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall?#embedding_9/StatefulPartitionedCall?lstm_1/StatefulPartitionedCall?
#embedding_9/StatefulPartitionedCallStatefulPartitionedCallembedding_9_inputembedding_9_162123*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_9_layer_call_and_return_conditional_losses_161273?
lstm_1/StatefulPartitionedCallStatefulPartitionedCall,embedding_9/StatefulPartitionedCall:output:0lstm_1_162126lstm_1_162128lstm_1_162130*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_161519?
 dense_18/StatefulPartitionedCallStatefulPartitionedCall'lstm_1/StatefulPartitionedCall:output:0dense_18_162133dense_18_162135*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_161538?
dropout_9/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_161549?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0dense_19_162139dense_19_162141*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_161562x
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall$^embedding_9/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????????????: : : : : : : : 2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2J
#embedding_9/StatefulPartitionedCall#embedding_9/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall:c _
0
_output_shapes
:??????????????????
+
_user_specified_nameembedding_9_input
?t
?	
while_body_161385
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
1while_lstm_cell_1_split_readvariableop_resource_0: @A
3while_lstm_cell_1_split_1_readvariableop_resource_0:@=
+while_lstm_cell_1_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
/while_lstm_cell_1_split_readvariableop_resource: @?
1while_lstm_cell_1_split_1_readvariableop_resource:@;
)while_lstm_cell_1_readvariableop_resource:@?? while/lstm_cell_1/ReadVariableOp?"while/lstm_cell_1/ReadVariableOp_1?"while/lstm_cell_1/ReadVariableOp_2?"while/lstm_cell_1/ReadVariableOp_3?&while/lstm_cell_1/split/ReadVariableOp?(while/lstm_cell_1/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype0?
!while/lstm_cell_1/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/ones_likeFill*while/lstm_cell_1/ones_like/Shape:output:0*while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? f
#while/lstm_cell_1/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:h
#while/lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/ones_like_1Fill,while/lstm_cell_1/ones_like_1/Shape:output:0,while/lstm_cell_1/ones_like_1/Const:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_1/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_1/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_1/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? c
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_1/split/ReadVariableOpReadVariableOp1while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes

: @*
dtype0?
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0.while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split?
while/lstm_cell_1/MatMulMatMulwhile/lstm_cell_1/mul:z:0 while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/MatMul_1MatMulwhile/lstm_cell_1/mul_1:z:0 while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/MatMul_2MatMulwhile/lstm_cell_1/mul_2:z:0 while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/MatMul_3MatMulwhile/lstm_cell_1/mul_3:z:0 while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????e
#while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
while/lstm_cell_1/split_1Split,while/lstm_cell_1/split_1/split_dim:output:00while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
while/lstm_cell_1/BiasAddBiasAdd"while/lstm_cell_1/MatMul:product:0"while/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/BiasAdd_1BiasAdd$while/lstm_cell_1/MatMul_1:product:0"while/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/BiasAdd_2BiasAdd$while/lstm_cell_1/MatMul_2:product:0"while/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/BiasAdd_3BiasAdd$while/lstm_cell_1/MatMul_3:product:0"while/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_4Mulwhile_placeholder_2&while/lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_5Mulwhile_placeholder_2&while/lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_6Mulwhile_placeholder_2&while/lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_7Mulwhile_placeholder_2&while/lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_1/ReadVariableOpReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0v
%while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_1/strided_sliceStridedSlice(while/lstm_cell_1/ReadVariableOp:value:0.while/lstm_cell_1/strided_slice/stack:output:00while/lstm_cell_1/strided_slice/stack_1:output:00while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_1/MatMul_4MatMulwhile/lstm_cell_1/mul_4:z:0(while/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/addAddV2"while/lstm_cell_1/BiasAdd:output:0$while/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????q
while/lstm_cell_1/SigmoidSigmoidwhile/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
"while/lstm_cell_1/ReadVariableOp_1ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0x
'while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_1/strided_slice_1StridedSlice*while/lstm_cell_1/ReadVariableOp_1:value:00while/lstm_cell_1/strided_slice_1/stack:output:02while/lstm_cell_1/strided_slice_1/stack_1:output:02while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_1/MatMul_5MatMulwhile/lstm_cell_1/mul_5:z:0*while/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/add_1AddV2$while/lstm_cell_1/BiasAdd_1:output:0$while/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????u
while/lstm_cell_1/Sigmoid_1Sigmoidwhile/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_8Mulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:??????????
"while/lstm_cell_1/ReadVariableOp_2ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0x
'while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   z
)while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_1/strided_slice_2StridedSlice*while/lstm_cell_1/ReadVariableOp_2:value:00while/lstm_cell_1/strided_slice_2/stack:output:02while/lstm_cell_1/strided_slice_2/stack_1:output:02while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_1/MatMul_6MatMulwhile/lstm_cell_1/mul_6:z:0*while/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/add_2AddV2$while/lstm_cell_1/BiasAdd_2:output:0$while/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????m
while/lstm_cell_1/TanhTanhwhile/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_9Mulwhile/lstm_cell_1/Sigmoid:y:0while/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/add_3AddV2while/lstm_cell_1/mul_8:z:0while/lstm_cell_1/mul_9:z:0*
T0*'
_output_shapes
:??????????
"while/lstm_cell_1/ReadVariableOp_3ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0x
'while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   z
)while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_1/strided_slice_3StridedSlice*while/lstm_cell_1/ReadVariableOp_3:value:00while/lstm_cell_1/strided_slice_3/stack:output:02while/lstm_cell_1/strided_slice_3/stack_1:output:02while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_1/MatMul_7MatMulwhile/lstm_cell_1/mul_7:z:0*while/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/add_4AddV2$while/lstm_cell_1/BiasAdd_3:output:0$while/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????u
while/lstm_cell_1/Sigmoid_2Sigmoidwhile/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????o
while/lstm_cell_1/Tanh_1Tanhwhile/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_10Mulwhile/lstm_cell_1/Sigmoid_2:y:0while/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_10:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_1/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:?????????x
while/Identity_5Identitywhile/lstm_cell_1/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_1_readvariableop_resource+while_lstm_cell_1_readvariableop_resource_0"h
1while_lstm_cell_1_split_1_readvariableop_resource3while_lstm_cell_1_split_1_readvariableop_resource_0"d
/while_lstm_cell_1_split_readvariableop_resource1while_lstm_cell_1_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????:?????????: : : : : 2D
 while/lstm_cell_1/ReadVariableOp while/lstm_cell_1/ReadVariableOp2H
"while/lstm_cell_1/ReadVariableOp_1"while/lstm_cell_1/ReadVariableOp_12H
"while/lstm_cell_1/ReadVariableOp_2"while/lstm_cell_1/ReadVariableOp_22H
"while/lstm_cell_1/ReadVariableOp_3"while/lstm_cell_1/ReadVariableOp_32P
&while/lstm_cell_1/split/ReadVariableOp&while/lstm_cell_1/split/ReadVariableOp2T
(while/lstm_cell_1/split_1/ReadVariableOp(while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
'__inference_lstm_1_layer_call_fn_162954

inputs
unknown: @
	unknown_0:@
	unknown_1:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_161519o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????????????? : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
ƅ
?
"__inference__traced_restore_164753
file_prefix:
'assignvariableop_embedding_9_embeddings:	?' 4
"assignvariableop_1_dense_18_kernel:
.
 assignvariableop_2_dense_18_bias:
4
"assignvariableop_3_dense_19_kernel:
.
 assignvariableop_4_dense_19_bias:&
assignvariableop_5_adam_iter:	 (
assignvariableop_6_adam_beta_1: (
assignvariableop_7_adam_beta_2: '
assignvariableop_8_adam_decay: /
%assignvariableop_9_adam_learning_rate: ?
-assignvariableop_10_lstm_1_lstm_cell_1_kernel: @I
7assignvariableop_11_lstm_1_lstm_cell_1_recurrent_kernel:@9
+assignvariableop_12_lstm_1_lstm_cell_1_bias:@#
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: D
1assignvariableop_17_adam_embedding_9_embeddings_m:	?' <
*assignvariableop_18_adam_dense_18_kernel_m:
6
(assignvariableop_19_adam_dense_18_bias_m:
<
*assignvariableop_20_adam_dense_19_kernel_m:
6
(assignvariableop_21_adam_dense_19_bias_m:F
4assignvariableop_22_adam_lstm_1_lstm_cell_1_kernel_m: @P
>assignvariableop_23_adam_lstm_1_lstm_cell_1_recurrent_kernel_m:@@
2assignvariableop_24_adam_lstm_1_lstm_cell_1_bias_m:@D
1assignvariableop_25_adam_embedding_9_embeddings_v:	?' <
*assignvariableop_26_adam_dense_18_kernel_v:
6
(assignvariableop_27_adam_dense_18_bias_v:
<
*assignvariableop_28_adam_dense_19_kernel_v:
6
(assignvariableop_29_adam_dense_19_bias_v:F
4assignvariableop_30_adam_lstm_1_lstm_cell_1_kernel_v: @P
>assignvariableop_31_adam_lstm_1_lstm_cell_1_recurrent_kernel_v:@@
2assignvariableop_32_adam_lstm_1_lstm_cell_1_bias_v:@
identity_34??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp'assignvariableop_embedding_9_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_18_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp assignvariableop_2_dense_18_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_19_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense_19_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp-assignvariableop_10_lstm_1_lstm_cell_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp7assignvariableop_11_lstm_1_lstm_cell_1_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp+assignvariableop_12_lstm_1_lstm_cell_1_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp1assignvariableop_17_adam_embedding_9_embeddings_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_dense_18_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_dense_18_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_dense_19_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_dense_19_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp4assignvariableop_22_adam_lstm_1_lstm_cell_1_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp>assignvariableop_23_adam_lstm_1_lstm_cell_1_recurrent_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp2assignvariableop_24_adam_lstm_1_lstm_cell_1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp1assignvariableop_25_adam_embedding_9_embeddings_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_dense_18_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_dense_18_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_dense_19_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_dense_19_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp4assignvariableop_30_adam_lstm_1_lstm_cell_1_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp>assignvariableop_31_adam_lstm_1_lstm_cell_1_recurrent_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp2assignvariableop_32_adam_lstm_1_lstm_cell_1_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_34Identity_34:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_162170
embedding_9_input%
embedding_9_162148:	?' 
lstm_1_162151: @
lstm_1_162153:@
lstm_1_162155:@!
dense_18_162158:

dense_18_162160:
!
dense_19_162164:

dense_19_162166:
identity?? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall?!dropout_9/StatefulPartitionedCall?#embedding_9/StatefulPartitionedCall?lstm_1/StatefulPartitionedCall?
#embedding_9/StatefulPartitionedCallStatefulPartitionedCallembedding_9_inputembedding_9_162148*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_9_layer_call_and_return_conditional_losses_161273?
lstm_1/StatefulPartitionedCallStatefulPartitionedCall,embedding_9/StatefulPartitionedCall:output:0lstm_1_162151lstm_1_162153lstm_1_162155*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_162015?
 dense_18/StatefulPartitionedCallStatefulPartitionedCall'lstm_1/StatefulPartitionedCall:output:0dense_18_162158dense_18_162160*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_161538?
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_161618?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0dense_19_162164dense_19_162166*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_161562x
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall$^embedding_9/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????????????: : : : : : : : 2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2J
#embedding_9/StatefulPartitionedCall#embedding_9/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall:c _
0
_output_shapes
:??????????????????
+
_user_specified_nameembedding_9_input
?	
?
G__inference_embedding_9_layer_call_and_return_conditional_losses_161273

inputs*
embedding_lookup_161267:	?' 
identity??embedding_lookup^
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:???????????????????
embedding_lookupResourceGatherembedding_lookup_161267Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/161267*4
_output_shapes"
 :?????????????????? *
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/161267*4
_output_shapes"
 :?????????????????? ?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :?????????????????? ?
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????????????: 2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?	
?
G__inference_embedding_9_layer_call_and_return_conditional_losses_162921

inputs*
embedding_lookup_162915:	?' 
identity??embedding_lookup^
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:???????????????????
embedding_lookupResourceGatherembedding_lookup_162915Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/162915*4
_output_shapes"
 :?????????????????? *
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/162915*4
_output_shapes"
 :?????????????????? ?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :?????????????????? ?
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????????????: 2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
??
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_162505

inputs6
#embedding_9_embedding_lookup_162245:	?' B
0lstm_1_lstm_cell_1_split_readvariableop_resource: @@
2lstm_1_lstm_cell_1_split_1_readvariableop_resource:@<
*lstm_1_lstm_cell_1_readvariableop_resource:@9
'dense_18_matmul_readvariableop_resource:
6
(dense_18_biasadd_readvariableop_resource:
9
'dense_19_matmul_readvariableop_resource:
6
(dense_19_biasadd_readvariableop_resource:
identity??dense_18/BiasAdd/ReadVariableOp?dense_18/MatMul/ReadVariableOp?dense_19/BiasAdd/ReadVariableOp?dense_19/MatMul/ReadVariableOp?embedding_9/embedding_lookup?!lstm_1/lstm_cell_1/ReadVariableOp?#lstm_1/lstm_cell_1/ReadVariableOp_1?#lstm_1/lstm_cell_1/ReadVariableOp_2?#lstm_1/lstm_cell_1/ReadVariableOp_3?'lstm_1/lstm_cell_1/split/ReadVariableOp?)lstm_1/lstm_cell_1/split_1/ReadVariableOp?lstm_1/whilej
embedding_9/CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:???????????????????
embedding_9/embedding_lookupResourceGather#embedding_9_embedding_lookup_162245embedding_9/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding_9/embedding_lookup/162245*4
_output_shapes"
 :?????????????????? *
dtype0?
%embedding_9/embedding_lookup/IdentityIdentity%embedding_9/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_9/embedding_lookup/162245*4
_output_shapes"
 :?????????????????? ?
'embedding_9/embedding_lookup/Identity_1Identity.embedding_9/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :?????????????????? l
lstm_1/ShapeShape0embedding_9/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:d
lstm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_1/strided_sliceStridedSlicelstm_1/Shape:output:0#lstm_1/strided_slice/stack:output:0%lstm_1/strided_slice/stack_1:output:0%lstm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
lstm_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :?
lstm_1/zeros/packedPacklstm_1/strided_slice:output:0lstm_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_1/zerosFilllstm_1/zeros/packed:output:0lstm_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????Y
lstm_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :?
lstm_1/zeros_1/packedPacklstm_1/strided_slice:output:0 lstm_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_1/zeros_1Filllstm_1/zeros_1/packed:output:0lstm_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????j
lstm_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_1/transpose	Transpose0embedding_9/embedding_lookup/Identity_1:output:0lstm_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? R
lstm_1/Shape_1Shapelstm_1/transpose:y:0*
T0*
_output_shapes
:f
lstm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_1/strided_slice_1StridedSlicelstm_1/Shape_1:output:0%lstm_1/strided_slice_1/stack:output:0'lstm_1/strided_slice_1/stack_1:output:0'lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
lstm_1/TensorArrayV2TensorListReserve+lstm_1/TensorArrayV2/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
.lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_1/transpose:y:0Elstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???f
lstm_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_1/strided_slice_2StridedSlicelstm_1/transpose:y:0%lstm_1/strided_slice_2/stack:output:0'lstm_1/strided_slice_2/stack_1:output:0'lstm_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_maskq
"lstm_1/lstm_cell_1/ones_like/ShapeShapelstm_1/strided_slice_2:output:0*
T0*
_output_shapes
:g
"lstm_1/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_1/lstm_cell_1/ones_likeFill+lstm_1/lstm_cell_1/ones_like/Shape:output:0+lstm_1/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? i
$lstm_1/lstm_cell_1/ones_like_1/ShapeShapelstm_1/zeros:output:0*
T0*
_output_shapes
:i
$lstm_1/lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_1/lstm_cell_1/ones_like_1Fill-lstm_1/lstm_cell_1/ones_like_1/Shape:output:0-lstm_1/lstm_cell_1/ones_like_1/Const:output:0*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/mulMullstm_1/strided_slice_2:output:0%lstm_1/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
lstm_1/lstm_cell_1/mul_1Mullstm_1/strided_slice_2:output:0%lstm_1/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
lstm_1/lstm_cell_1/mul_2Mullstm_1/strided_slice_2:output:0%lstm_1/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
lstm_1/lstm_cell_1/mul_3Mullstm_1/strided_slice_2:output:0%lstm_1/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? d
"lstm_1/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
'lstm_1/lstm_cell_1/split/ReadVariableOpReadVariableOp0lstm_1_lstm_cell_1_split_readvariableop_resource*
_output_shapes

: @*
dtype0?
lstm_1/lstm_cell_1/splitSplit+lstm_1/lstm_cell_1/split/split_dim:output:0/lstm_1/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split?
lstm_1/lstm_cell_1/MatMulMatMullstm_1/lstm_cell_1/mul:z:0!lstm_1/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/MatMul_1MatMullstm_1/lstm_cell_1/mul_1:z:0!lstm_1/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/MatMul_2MatMullstm_1/lstm_cell_1/mul_2:z:0!lstm_1/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/MatMul_3MatMullstm_1/lstm_cell_1/mul_3:z:0!lstm_1/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????f
$lstm_1/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
)lstm_1/lstm_cell_1/split_1/ReadVariableOpReadVariableOp2lstm_1_lstm_cell_1_split_1_readvariableop_resource*
_output_shapes
:@*
dtype0?
lstm_1/lstm_cell_1/split_1Split-lstm_1/lstm_cell_1/split_1/split_dim:output:01lstm_1/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
lstm_1/lstm_cell_1/BiasAddBiasAdd#lstm_1/lstm_cell_1/MatMul:product:0#lstm_1/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/BiasAdd_1BiasAdd%lstm_1/lstm_cell_1/MatMul_1:product:0#lstm_1/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/BiasAdd_2BiasAdd%lstm_1/lstm_cell_1/MatMul_2:product:0#lstm_1/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/BiasAdd_3BiasAdd%lstm_1/lstm_cell_1/MatMul_3:product:0#lstm_1/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/mul_4Mullstm_1/zeros:output:0'lstm_1/lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/mul_5Mullstm_1/zeros:output:0'lstm_1/lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/mul_6Mullstm_1/zeros:output:0'lstm_1/lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/mul_7Mullstm_1/zeros:output:0'lstm_1/lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
!lstm_1/lstm_cell_1/ReadVariableOpReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0w
&lstm_1/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(lstm_1/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(lstm_1/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
 lstm_1/lstm_cell_1/strided_sliceStridedSlice)lstm_1/lstm_cell_1/ReadVariableOp:value:0/lstm_1/lstm_cell_1/strided_slice/stack:output:01lstm_1/lstm_cell_1/strided_slice/stack_1:output:01lstm_1/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_1/lstm_cell_1/MatMul_4MatMullstm_1/lstm_cell_1/mul_4:z:0)lstm_1/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/addAddV2#lstm_1/lstm_cell_1/BiasAdd:output:0%lstm_1/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????s
lstm_1/lstm_cell_1/SigmoidSigmoidlstm_1/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
#lstm_1/lstm_cell_1/ReadVariableOp_1ReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0y
(lstm_1/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_1/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_1/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_1/lstm_cell_1/strided_slice_1StridedSlice+lstm_1/lstm_cell_1/ReadVariableOp_1:value:01lstm_1/lstm_cell_1/strided_slice_1/stack:output:03lstm_1/lstm_cell_1/strided_slice_1/stack_1:output:03lstm_1/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_1/lstm_cell_1/MatMul_5MatMullstm_1/lstm_cell_1/mul_5:z:0+lstm_1/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/add_1AddV2%lstm_1/lstm_cell_1/BiasAdd_1:output:0%lstm_1/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????w
lstm_1/lstm_cell_1/Sigmoid_1Sigmoidlstm_1/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/mul_8Mul lstm_1/lstm_cell_1/Sigmoid_1:y:0lstm_1/zeros_1:output:0*
T0*'
_output_shapes
:??????????
#lstm_1/lstm_cell_1/ReadVariableOp_2ReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0y
(lstm_1/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_1/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   {
*lstm_1/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_1/lstm_cell_1/strided_slice_2StridedSlice+lstm_1/lstm_cell_1/ReadVariableOp_2:value:01lstm_1/lstm_cell_1/strided_slice_2/stack:output:03lstm_1/lstm_cell_1/strided_slice_2/stack_1:output:03lstm_1/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_1/lstm_cell_1/MatMul_6MatMullstm_1/lstm_cell_1/mul_6:z:0+lstm_1/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/add_2AddV2%lstm_1/lstm_cell_1/BiasAdd_2:output:0%lstm_1/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????o
lstm_1/lstm_cell_1/TanhTanhlstm_1/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/mul_9Mullstm_1/lstm_cell_1/Sigmoid:y:0lstm_1/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/add_3AddV2lstm_1/lstm_cell_1/mul_8:z:0lstm_1/lstm_cell_1/mul_9:z:0*
T0*'
_output_shapes
:??????????
#lstm_1/lstm_cell_1/ReadVariableOp_3ReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0y
(lstm_1/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   {
*lstm_1/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_1/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_1/lstm_cell_1/strided_slice_3StridedSlice+lstm_1/lstm_cell_1/ReadVariableOp_3:value:01lstm_1/lstm_cell_1/strided_slice_3/stack:output:03lstm_1/lstm_cell_1/strided_slice_3/stack_1:output:03lstm_1/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_1/lstm_cell_1/MatMul_7MatMullstm_1/lstm_cell_1/mul_7:z:0+lstm_1/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/add_4AddV2%lstm_1/lstm_cell_1/BiasAdd_3:output:0%lstm_1/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????w
lstm_1/lstm_cell_1/Sigmoid_2Sigmoidlstm_1/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????q
lstm_1/lstm_cell_1/Tanh_1Tanhlstm_1/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/mul_10Mul lstm_1/lstm_cell_1/Sigmoid_2:y:0lstm_1/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????u
$lstm_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
lstm_1/TensorArrayV2_1TensorListReserve-lstm_1/TensorArrayV2_1/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???M
lstm_1/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????[
lstm_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
lstm_1/whileWhile"lstm_1/while/loop_counter:output:0(lstm_1/while/maximum_iterations:output:0lstm_1/time:output:0lstm_1/TensorArrayV2_1:handle:0lstm_1/zeros:output:0lstm_1/zeros_1:output:0lstm_1/strided_slice_1:output:0>lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:00lstm_1_lstm_cell_1_split_readvariableop_resource2lstm_1_lstm_cell_1_split_1_readvariableop_resource*lstm_1_lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_1_while_body_162356*$
condR
lstm_1_while_cond_162355*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations ?
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)lstm_1/TensorArrayV2Stack/TensorListStackTensorListStacklstm_1/while:output:3@lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0o
lstm_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????h
lstm_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_1/strided_slice_3StridedSlice2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_1/strided_slice_3/stack:output:0'lstm_1/strided_slice_3/stack_1:output:0'lstm_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskl
lstm_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_1/transpose_1	Transpose2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_1/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????b
lstm_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ?
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
dense_18/MatMulMatMullstm_1/strided_slice_3:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
b
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
m
dropout_9/IdentityIdentitydense_18/Relu:activations:0*
T0*'
_output_shapes
:?????????
?
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
dense_19/MatMulMatMuldropout_9/Identity:output:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_19/SigmoidSigmoiddense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
IdentityIdentitydense_19/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp^embedding_9/embedding_lookup"^lstm_1/lstm_cell_1/ReadVariableOp$^lstm_1/lstm_cell_1/ReadVariableOp_1$^lstm_1/lstm_cell_1/ReadVariableOp_2$^lstm_1/lstm_cell_1/ReadVariableOp_3(^lstm_1/lstm_cell_1/split/ReadVariableOp*^lstm_1/lstm_cell_1/split_1/ReadVariableOp^lstm_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????????????: : : : : : : : 2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2<
embedding_9/embedding_lookupembedding_9/embedding_lookup2F
!lstm_1/lstm_cell_1/ReadVariableOp!lstm_1/lstm_cell_1/ReadVariableOp2J
#lstm_1/lstm_cell_1/ReadVariableOp_1#lstm_1/lstm_cell_1/ReadVariableOp_12J
#lstm_1/lstm_cell_1/ReadVariableOp_2#lstm_1/lstm_cell_1/ReadVariableOp_22J
#lstm_1/lstm_cell_1/ReadVariableOp_3#lstm_1/lstm_cell_1/ReadVariableOp_32R
'lstm_1/lstm_cell_1/split/ReadVariableOp'lstm_1/lstm_cell_1/split/ReadVariableOp2V
)lstm_1/lstm_cell_1/split_1/ReadVariableOp)lstm_1/lstm_cell_1/split_1/ReadVariableOp2
lstm_1/whilelstm_1/while:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?	
?
-__inference_sequential_9_layer_call_fn_162241

inputs
unknown:	?' 
	unknown_0: @
	unknown_1:@
	unknown_2:@
	unknown_3:

	unknown_4:

	unknown_5:

	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_162080o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
??
?
B__inference_lstm_1_layer_call_and_return_conditional_losses_163822

inputs;
)lstm_cell_1_split_readvariableop_resource: @9
+lstm_cell_1_split_1_readvariableop_resource:@5
#lstm_cell_1_readvariableop_resource:@
identity??lstm_cell_1/ReadVariableOp?lstm_cell_1/ReadVariableOp_1?lstm_cell_1/ReadVariableOp_2?lstm_cell_1/ReadVariableOp_3? lstm_cell_1/split/ReadVariableOp?"lstm_cell_1/split_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_maskc
lstm_cell_1/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:`
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? [
lstm_cell_1/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:b
lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/ones_like_1Fill&lstm_cell_1/ones_like_1/Shape:output:0&lstm_cell_1/ones_like_1/Const:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/mulMulstrided_slice_2:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/mul_1Mulstrided_slice_2:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/mul_2Mulstrided_slice_2:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/mul_3Mulstrided_slice_2:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_1/split/ReadVariableOpReadVariableOp)lstm_cell_1_split_readvariableop_resource*
_output_shapes

: @*
dtype0?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0(lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split
lstm_cell_1/MatMulMatMullstm_cell_1/mul:z:0lstm_cell_1/split:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/MatMul_1MatMullstm_cell_1/mul_1:z:0lstm_cell_1/split:output:1*
T0*'
_output_shapes
:??????????
lstm_cell_1/MatMul_2MatMullstm_cell_1/mul_2:z:0lstm_cell_1/split:output:2*
T0*'
_output_shapes
:??????????
lstm_cell_1/MatMul_3MatMullstm_cell_1/mul_3:z:0lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????_
lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_1/split_1/ReadVariableOpReadVariableOp+lstm_cell_1_split_1_readvariableop_resource*
_output_shapes
:@*
dtype0?
lstm_cell_1/split_1Split&lstm_cell_1/split_1/split_dim:output:0*lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/MatMul:product:0lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/BiasAdd_1BiasAddlstm_cell_1/MatMul_1:product:0lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:??????????
lstm_cell_1/BiasAdd_2BiasAddlstm_cell_1/MatMul_2:product:0lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
lstm_cell_1/BiasAdd_3BiasAddlstm_cell_1/MatMul_3:product:0lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????|
lstm_cell_1/mul_4Mulzeros:output:0 lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:?????????|
lstm_cell_1/mul_5Mulzeros:output:0 lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:?????????|
lstm_cell_1/mul_6Mulzeros:output:0 lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:?????????|
lstm_cell_1/mul_7Mulzeros:output:0 lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:?????????~
lstm_cell_1/ReadVariableOpReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0p
lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_1/strided_sliceStridedSlice"lstm_cell_1/ReadVariableOp:value:0(lstm_cell_1/strided_slice/stack:output:0*lstm_cell_1/strided_slice/stack_1:output:0*lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_1/MatMul_4MatMullstm_cell_1/mul_4:z:0"lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/addAddV2lstm_cell_1/BiasAdd:output:0lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????e
lstm_cell_1/SigmoidSigmoidlstm_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/ReadVariableOp_1ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0r
!lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_1/strided_slice_1StridedSlice$lstm_cell_1/ReadVariableOp_1:value:0*lstm_cell_1/strided_slice_1/stack:output:0,lstm_cell_1/strided_slice_1/stack_1:output:0,lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_1/MatMul_5MatMullstm_cell_1/mul_5:z:0$lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/add_1AddV2lstm_cell_1/BiasAdd_1:output:0lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????i
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????w
lstm_cell_1/mul_8Mullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/ReadVariableOp_2ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0r
!lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   t
#lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_1/strided_slice_2StridedSlice$lstm_cell_1/ReadVariableOp_2:value:0*lstm_cell_1/strided_slice_2/stack:output:0,lstm_cell_1/strided_slice_2/stack_1:output:0,lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_1/MatMul_6MatMullstm_cell_1/mul_6:z:0$lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/add_2AddV2lstm_cell_1/BiasAdd_2:output:0lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????a
lstm_cell_1/TanhTanhlstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????y
lstm_cell_1/mul_9Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????z
lstm_cell_1/add_3AddV2lstm_cell_1/mul_8:z:0lstm_cell_1/mul_9:z:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/ReadVariableOp_3ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0r
!lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   t
#lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_1/strided_slice_3StridedSlice$lstm_cell_1/ReadVariableOp_3:value:0*lstm_cell_1/strided_slice_3/stack:output:0,lstm_cell_1/strided_slice_3/stack_1:output:0,lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_1/MatMul_7MatMullstm_cell_1/mul_7:z:0$lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/add_4AddV2lstm_cell_1/BiasAdd_3:output:0lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????i
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????c
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????~
lstm_cell_1/mul_10Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_1_split_readvariableop_resource+lstm_cell_1_split_1_readvariableop_resource#lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_163688*
condR
while_cond_163687*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????????????? : : : 28
lstm_cell_1/ReadVariableOplstm_cell_1/ReadVariableOp2<
lstm_cell_1/ReadVariableOp_1lstm_cell_1/ReadVariableOp_12<
lstm_cell_1/ReadVariableOp_2lstm_cell_1/ReadVariableOp_22<
lstm_cell_1/ReadVariableOp_3lstm_cell_1/ReadVariableOp_32D
 lstm_cell_1/split/ReadVariableOp lstm_cell_1/split/ReadVariableOp2H
"lstm_cell_1/split_1/ReadVariableOp"lstm_cell_1/split_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?	
?
lstm_1_while_cond_162355*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3,
(lstm_1_while_less_lstm_1_strided_slice_1B
>lstm_1_while_lstm_1_while_cond_162355___redundant_placeholder0B
>lstm_1_while_lstm_1_while_cond_162355___redundant_placeholder1B
>lstm_1_while_lstm_1_while_cond_162355___redundant_placeholder2B
>lstm_1_while_lstm_1_while_cond_162355___redundant_placeholder3
lstm_1_while_identity
~
lstm_1/while/LessLesslstm_1_while_placeholder(lstm_1_while_less_lstm_1_strided_slice_1*
T0*
_output_shapes
: Y
lstm_1/while/IdentityIdentitylstm_1/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_1_while_identitylstm_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
??
?	
while_body_163995
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
1while_lstm_cell_1_split_readvariableop_resource_0: @A
3while_lstm_cell_1_split_1_readvariableop_resource_0:@=
+while_lstm_cell_1_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
/while_lstm_cell_1_split_readvariableop_resource: @?
1while_lstm_cell_1_split_1_readvariableop_resource:@;
)while_lstm_cell_1_readvariableop_resource:@?? while/lstm_cell_1/ReadVariableOp?"while/lstm_cell_1/ReadVariableOp_1?"while/lstm_cell_1/ReadVariableOp_2?"while/lstm_cell_1/ReadVariableOp_3?&while/lstm_cell_1/split/ReadVariableOp?(while/lstm_cell_1/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype0?
!while/lstm_cell_1/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/ones_likeFill*while/lstm_cell_1/ones_like/Shape:output:0*while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? d
while/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/dropout/MulMul$while/lstm_cell_1/ones_like:output:0(while/lstm_cell_1/dropout/Const:output:0*
T0*'
_output_shapes
:????????? s
while/lstm_cell_1/dropout/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
6while/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_1/dropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2??"m
(while/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
&while/lstm_cell_1/dropout/GreaterEqualGreaterEqual?while/lstm_cell_1/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_1/dropout/CastCast*while/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
while/lstm_cell_1/dropout/Mul_1Mul!while/lstm_cell_1/dropout/Mul:z:0"while/lstm_cell_1/dropout/Cast:y:0*
T0*'
_output_shapes
:????????? f
!while/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/dropout_1/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_1/Const:output:0*
T0*'
_output_shapes
:????????? u
!while/lstm_cell_1/dropout_1/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_1/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2??o
*while/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
 while/lstm_cell_1/dropout_1/CastCast,while/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
!while/lstm_cell_1/dropout_1/Mul_1Mul#while/lstm_cell_1/dropout_1/Mul:z:0$while/lstm_cell_1/dropout_1/Cast:y:0*
T0*'
_output_shapes
:????????? f
!while/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/dropout_2/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_2/Const:output:0*
T0*'
_output_shapes
:????????? u
!while/lstm_cell_1/dropout_2/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_2/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???o
*while/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
 while/lstm_cell_1/dropout_2/CastCast,while/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
!while/lstm_cell_1/dropout_2/Mul_1Mul#while/lstm_cell_1/dropout_2/Mul:z:0$while/lstm_cell_1/dropout_2/Cast:y:0*
T0*'
_output_shapes
:????????? f
!while/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/dropout_3/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_3/Const:output:0*
T0*'
_output_shapes
:????????? u
!while/lstm_cell_1/dropout_3/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_3/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2ۏ?o
*while/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
 while/lstm_cell_1/dropout_3/CastCast,while/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
!while/lstm_cell_1/dropout_3/Mul_1Mul#while/lstm_cell_1/dropout_3/Mul:z:0$while/lstm_cell_1/dropout_3/Cast:y:0*
T0*'
_output_shapes
:????????? f
#while/lstm_cell_1/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:h
#while/lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/ones_like_1Fill,while/lstm_cell_1/ones_like_1/Shape:output:0,while/lstm_cell_1/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????f
!while/lstm_cell_1/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/dropout_4/MulMul&while/lstm_cell_1/ones_like_1:output:0*while/lstm_cell_1/dropout_4/Const:output:0*
T0*'
_output_shapes
:?????????w
!while/lstm_cell_1/dropout_4/ShapeShape&while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_1/dropout_4/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???o
*while/lstm_cell_1/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_1/dropout_4/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_4/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_1/dropout_4/CastCast,while/lstm_cell_1/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
!while/lstm_cell_1/dropout_4/Mul_1Mul#while/lstm_cell_1/dropout_4/Mul:z:0$while/lstm_cell_1/dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????f
!while/lstm_cell_1/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/dropout_5/MulMul&while/lstm_cell_1/ones_like_1:output:0*while/lstm_cell_1/dropout_5/Const:output:0*
T0*'
_output_shapes
:?????????w
!while/lstm_cell_1/dropout_5/ShapeShape&while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_1/dropout_5/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???o
*while/lstm_cell_1/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_1/dropout_5/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_5/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_1/dropout_5/CastCast,while/lstm_cell_1/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
!while/lstm_cell_1/dropout_5/Mul_1Mul#while/lstm_cell_1/dropout_5/Mul:z:0$while/lstm_cell_1/dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????f
!while/lstm_cell_1/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/dropout_6/MulMul&while/lstm_cell_1/ones_like_1:output:0*while/lstm_cell_1/dropout_6/Const:output:0*
T0*'
_output_shapes
:?????????w
!while/lstm_cell_1/dropout_6/ShapeShape&while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_1/dropout_6/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_6/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???o
*while/lstm_cell_1/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_1/dropout_6/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_6/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_1/dropout_6/CastCast,while/lstm_cell_1/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
!while/lstm_cell_1/dropout_6/Mul_1Mul#while/lstm_cell_1/dropout_6/Mul:z:0$while/lstm_cell_1/dropout_6/Cast:y:0*
T0*'
_output_shapes
:?????????f
!while/lstm_cell_1/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/dropout_7/MulMul&while/lstm_cell_1/ones_like_1:output:0*while/lstm_cell_1/dropout_7/Const:output:0*
T0*'
_output_shapes
:?????????w
!while/lstm_cell_1/dropout_7/ShapeShape&while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_1/dropout_7/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_7/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???o
*while/lstm_cell_1/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_1/dropout_7/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_7/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_1/dropout_7/CastCast,while/lstm_cell_1/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
!while/lstm_cell_1/dropout_7/Mul_1Mul#while/lstm_cell_1/dropout_7/Mul:z:0$while/lstm_cell_1/dropout_7/Cast:y:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_1/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_1/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_1/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_1/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_1/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_1/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:????????? c
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_1/split/ReadVariableOpReadVariableOp1while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes

: @*
dtype0?
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0.while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split?
while/lstm_cell_1/MatMulMatMulwhile/lstm_cell_1/mul:z:0 while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/MatMul_1MatMulwhile/lstm_cell_1/mul_1:z:0 while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/MatMul_2MatMulwhile/lstm_cell_1/mul_2:z:0 while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/MatMul_3MatMulwhile/lstm_cell_1/mul_3:z:0 while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????e
#while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
while/lstm_cell_1/split_1Split,while/lstm_cell_1/split_1/split_dim:output:00while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
while/lstm_cell_1/BiasAddBiasAdd"while/lstm_cell_1/MatMul:product:0"while/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/BiasAdd_1BiasAdd$while/lstm_cell_1/MatMul_1:product:0"while/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/BiasAdd_2BiasAdd$while/lstm_cell_1/MatMul_2:product:0"while/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/BiasAdd_3BiasAdd$while/lstm_cell_1/MatMul_3:product:0"while/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_4Mulwhile_placeholder_2%while/lstm_cell_1/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_5Mulwhile_placeholder_2%while/lstm_cell_1/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_6Mulwhile_placeholder_2%while/lstm_cell_1/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_7Mulwhile_placeholder_2%while/lstm_cell_1/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_1/ReadVariableOpReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0v
%while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_1/strided_sliceStridedSlice(while/lstm_cell_1/ReadVariableOp:value:0.while/lstm_cell_1/strided_slice/stack:output:00while/lstm_cell_1/strided_slice/stack_1:output:00while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_1/MatMul_4MatMulwhile/lstm_cell_1/mul_4:z:0(while/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/addAddV2"while/lstm_cell_1/BiasAdd:output:0$while/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????q
while/lstm_cell_1/SigmoidSigmoidwhile/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
"while/lstm_cell_1/ReadVariableOp_1ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0x
'while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_1/strided_slice_1StridedSlice*while/lstm_cell_1/ReadVariableOp_1:value:00while/lstm_cell_1/strided_slice_1/stack:output:02while/lstm_cell_1/strided_slice_1/stack_1:output:02while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_1/MatMul_5MatMulwhile/lstm_cell_1/mul_5:z:0*while/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/add_1AddV2$while/lstm_cell_1/BiasAdd_1:output:0$while/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????u
while/lstm_cell_1/Sigmoid_1Sigmoidwhile/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_8Mulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:??????????
"while/lstm_cell_1/ReadVariableOp_2ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0x
'while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   z
)while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_1/strided_slice_2StridedSlice*while/lstm_cell_1/ReadVariableOp_2:value:00while/lstm_cell_1/strided_slice_2/stack:output:02while/lstm_cell_1/strided_slice_2/stack_1:output:02while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_1/MatMul_6MatMulwhile/lstm_cell_1/mul_6:z:0*while/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/add_2AddV2$while/lstm_cell_1/BiasAdd_2:output:0$while/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????m
while/lstm_cell_1/TanhTanhwhile/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_9Mulwhile/lstm_cell_1/Sigmoid:y:0while/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/add_3AddV2while/lstm_cell_1/mul_8:z:0while/lstm_cell_1/mul_9:z:0*
T0*'
_output_shapes
:??????????
"while/lstm_cell_1/ReadVariableOp_3ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0x
'while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   z
)while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_1/strided_slice_3StridedSlice*while/lstm_cell_1/ReadVariableOp_3:value:00while/lstm_cell_1/strided_slice_3/stack:output:02while/lstm_cell_1/strided_slice_3/stack_1:output:02while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_1/MatMul_7MatMulwhile/lstm_cell_1/mul_7:z:0*while/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/add_4AddV2$while/lstm_cell_1/BiasAdd_3:output:0$while/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????u
while/lstm_cell_1/Sigmoid_2Sigmoidwhile/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????o
while/lstm_cell_1/Tanh_1Tanhwhile/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_10Mulwhile/lstm_cell_1/Sigmoid_2:y:0while/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_10:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_1/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:?????????x
while/Identity_5Identitywhile/lstm_cell_1/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_1_readvariableop_resource+while_lstm_cell_1_readvariableop_resource_0"h
1while_lstm_cell_1_split_1_readvariableop_resource3while_lstm_cell_1_split_1_readvariableop_resource_0"d
/while_lstm_cell_1_split_readvariableop_resource1while_lstm_cell_1_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????:?????????: : : : : 2D
 while/lstm_cell_1/ReadVariableOp while/lstm_cell_1/ReadVariableOp2H
"while/lstm_cell_1/ReadVariableOp_1"while/lstm_cell_1/ReadVariableOp_12H
"while/lstm_cell_1/ReadVariableOp_2"while/lstm_cell_1/ReadVariableOp_22H
"while/lstm_cell_1/ReadVariableOp_3"while/lstm_cell_1/ReadVariableOp_32P
&while/lstm_cell_1/split/ReadVariableOp&while/lstm_cell_1/split/ReadVariableOp2T
(while/lstm_cell_1/split_1/ReadVariableOp(while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
)__inference_dense_18_layer_call_fn_164202

inputs
unknown:

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_161538o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
B__inference_lstm_1_layer_call_and_return_conditional_losses_162015

inputs;
)lstm_cell_1_split_readvariableop_resource: @9
+lstm_cell_1_split_1_readvariableop_resource:@5
#lstm_cell_1_readvariableop_resource:@
identity??lstm_cell_1/ReadVariableOp?lstm_cell_1/ReadVariableOp_1?lstm_cell_1/ReadVariableOp_2?lstm_cell_1/ReadVariableOp_3? lstm_cell_1/split/ReadVariableOp?"lstm_cell_1/split_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_maskc
lstm_cell_1/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:`
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? ^
lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/dropout/MulMullstm_cell_1/ones_like:output:0"lstm_cell_1/dropout/Const:output:0*
T0*'
_output_shapes
:????????? g
lstm_cell_1/dropout/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
0lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_1/dropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???g
"lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
 lstm_cell_1/dropout/GreaterEqualGreaterEqual9lstm_cell_1/dropout/random_uniform/RandomUniform:output:0+lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/dropout/CastCast$lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
lstm_cell_1/dropout/Mul_1Mullstm_cell_1/dropout/Mul:z:0lstm_cell_1/dropout/Cast:y:0*
T0*'
_output_shapes
:????????? `
lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/dropout_1/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_1/Const:output:0*
T0*'
_output_shapes
:????????? i
lstm_cell_1/dropout_1/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_1/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???i
$lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_1/dropout_1/GreaterEqualGreaterEqual;lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/dropout_1/CastCast&lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
lstm_cell_1/dropout_1/Mul_1Mullstm_cell_1/dropout_1/Mul:z:0lstm_cell_1/dropout_1/Cast:y:0*
T0*'
_output_shapes
:????????? `
lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/dropout_2/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_2/Const:output:0*
T0*'
_output_shapes
:????????? i
lstm_cell_1/dropout_2/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_2/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???i
$lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_1/dropout_2/GreaterEqualGreaterEqual;lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/dropout_2/CastCast&lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
lstm_cell_1/dropout_2/Mul_1Mullstm_cell_1/dropout_2/Mul:z:0lstm_cell_1/dropout_2/Cast:y:0*
T0*'
_output_shapes
:????????? `
lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/dropout_3/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_3/Const:output:0*
T0*'
_output_shapes
:????????? i
lstm_cell_1/dropout_3/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_3/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???i
$lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_1/dropout_3/GreaterEqualGreaterEqual;lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/dropout_3/CastCast&lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
lstm_cell_1/dropout_3/Mul_1Mullstm_cell_1/dropout_3/Mul:z:0lstm_cell_1/dropout_3/Cast:y:0*
T0*'
_output_shapes
:????????? [
lstm_cell_1/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:b
lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/ones_like_1Fill&lstm_cell_1/ones_like_1/Shape:output:0&lstm_cell_1/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????`
lstm_cell_1/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/dropout_4/MulMul lstm_cell_1/ones_like_1:output:0$lstm_cell_1/dropout_4/Const:output:0*
T0*'
_output_shapes
:?????????k
lstm_cell_1/dropout_4/ShapeShape lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_1/dropout_4/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2??}i
$lstm_cell_1/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_1/dropout_4/GreaterEqualGreaterEqual;lstm_cell_1/dropout_4/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/dropout_4/CastCast&lstm_cell_1/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
lstm_cell_1/dropout_4/Mul_1Mullstm_cell_1/dropout_4/Mul:z:0lstm_cell_1/dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????`
lstm_cell_1/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/dropout_5/MulMul lstm_cell_1/ones_like_1:output:0$lstm_cell_1/dropout_5/Const:output:0*
T0*'
_output_shapes
:?????????k
lstm_cell_1/dropout_5/ShapeShape lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_1/dropout_5/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2о?i
$lstm_cell_1/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_1/dropout_5/GreaterEqualGreaterEqual;lstm_cell_1/dropout_5/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/dropout_5/CastCast&lstm_cell_1/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
lstm_cell_1/dropout_5/Mul_1Mullstm_cell_1/dropout_5/Mul:z:0lstm_cell_1/dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????`
lstm_cell_1/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/dropout_6/MulMul lstm_cell_1/ones_like_1:output:0$lstm_cell_1/dropout_6/Const:output:0*
T0*'
_output_shapes
:?????????k
lstm_cell_1/dropout_6/ShapeShape lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_1/dropout_6/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_6/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???i
$lstm_cell_1/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_1/dropout_6/GreaterEqualGreaterEqual;lstm_cell_1/dropout_6/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/dropout_6/CastCast&lstm_cell_1/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
lstm_cell_1/dropout_6/Mul_1Mullstm_cell_1/dropout_6/Mul:z:0lstm_cell_1/dropout_6/Cast:y:0*
T0*'
_output_shapes
:?????????`
lstm_cell_1/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/dropout_7/MulMul lstm_cell_1/ones_like_1:output:0$lstm_cell_1/dropout_7/Const:output:0*
T0*'
_output_shapes
:?????????k
lstm_cell_1/dropout_7/ShapeShape lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_1/dropout_7/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_7/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2??Ii
$lstm_cell_1/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_1/dropout_7/GreaterEqualGreaterEqual;lstm_cell_1/dropout_7/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/dropout_7/CastCast&lstm_cell_1/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
lstm_cell_1/dropout_7/Mul_1Mullstm_cell_1/dropout_7/Mul:z:0lstm_cell_1/dropout_7/Cast:y:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/mulMulstrided_slice_2:output:0lstm_cell_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/mul_1Mulstrided_slice_2:output:0lstm_cell_1/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/mul_2Mulstrided_slice_2:output:0lstm_cell_1/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/mul_3Mulstrided_slice_2:output:0lstm_cell_1/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:????????? ]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_1/split/ReadVariableOpReadVariableOp)lstm_cell_1_split_readvariableop_resource*
_output_shapes

: @*
dtype0?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0(lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split
lstm_cell_1/MatMulMatMullstm_cell_1/mul:z:0lstm_cell_1/split:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/MatMul_1MatMullstm_cell_1/mul_1:z:0lstm_cell_1/split:output:1*
T0*'
_output_shapes
:??????????
lstm_cell_1/MatMul_2MatMullstm_cell_1/mul_2:z:0lstm_cell_1/split:output:2*
T0*'
_output_shapes
:??????????
lstm_cell_1/MatMul_3MatMullstm_cell_1/mul_3:z:0lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????_
lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_1/split_1/ReadVariableOpReadVariableOp+lstm_cell_1_split_1_readvariableop_resource*
_output_shapes
:@*
dtype0?
lstm_cell_1/split_1Split&lstm_cell_1/split_1/split_dim:output:0*lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/MatMul:product:0lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/BiasAdd_1BiasAddlstm_cell_1/MatMul_1:product:0lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:??????????
lstm_cell_1/BiasAdd_2BiasAddlstm_cell_1/MatMul_2:product:0lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
lstm_cell_1/BiasAdd_3BiasAddlstm_cell_1/MatMul_3:product:0lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????{
lstm_cell_1/mul_4Mulzeros:output:0lstm_cell_1/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:?????????{
lstm_cell_1/mul_5Mulzeros:output:0lstm_cell_1/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:?????????{
lstm_cell_1/mul_6Mulzeros:output:0lstm_cell_1/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:?????????{
lstm_cell_1/mul_7Mulzeros:output:0lstm_cell_1/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:?????????~
lstm_cell_1/ReadVariableOpReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0p
lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_1/strided_sliceStridedSlice"lstm_cell_1/ReadVariableOp:value:0(lstm_cell_1/strided_slice/stack:output:0*lstm_cell_1/strided_slice/stack_1:output:0*lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_1/MatMul_4MatMullstm_cell_1/mul_4:z:0"lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/addAddV2lstm_cell_1/BiasAdd:output:0lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????e
lstm_cell_1/SigmoidSigmoidlstm_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/ReadVariableOp_1ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0r
!lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_1/strided_slice_1StridedSlice$lstm_cell_1/ReadVariableOp_1:value:0*lstm_cell_1/strided_slice_1/stack:output:0,lstm_cell_1/strided_slice_1/stack_1:output:0,lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_1/MatMul_5MatMullstm_cell_1/mul_5:z:0$lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/add_1AddV2lstm_cell_1/BiasAdd_1:output:0lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????i
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????w
lstm_cell_1/mul_8Mullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/ReadVariableOp_2ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0r
!lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   t
#lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_1/strided_slice_2StridedSlice$lstm_cell_1/ReadVariableOp_2:value:0*lstm_cell_1/strided_slice_2/stack:output:0,lstm_cell_1/strided_slice_2/stack_1:output:0,lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_1/MatMul_6MatMullstm_cell_1/mul_6:z:0$lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/add_2AddV2lstm_cell_1/BiasAdd_2:output:0lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????a
lstm_cell_1/TanhTanhlstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????y
lstm_cell_1/mul_9Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????z
lstm_cell_1/add_3AddV2lstm_cell_1/mul_8:z:0lstm_cell_1/mul_9:z:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/ReadVariableOp_3ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0r
!lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   t
#lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_1/strided_slice_3StridedSlice$lstm_cell_1/ReadVariableOp_3:value:0*lstm_cell_1/strided_slice_3/stack:output:0,lstm_cell_1/strided_slice_3/stack_1:output:0,lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_1/MatMul_7MatMullstm_cell_1/mul_7:z:0$lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/add_4AddV2lstm_cell_1/BiasAdd_3:output:0lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????i
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????c
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????~
lstm_cell_1/mul_10Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_1_split_readvariableop_resource+lstm_cell_1_split_1_readvariableop_resource#lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_161817*
condR
while_cond_161816*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????????????? : : : 28
lstm_cell_1/ReadVariableOplstm_cell_1/ReadVariableOp2<
lstm_cell_1/ReadVariableOp_1lstm_cell_1/ReadVariableOp_12<
lstm_cell_1/ReadVariableOp_2lstm_cell_1/ReadVariableOp_22<
lstm_cell_1/ReadVariableOp_3lstm_cell_1/ReadVariableOp_32D
 lstm_cell_1/split/ReadVariableOp lstm_cell_1/split/ReadVariableOp2H
"lstm_cell_1/split_1/ReadVariableOp"lstm_cell_1/split_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?"
?
while_body_161178
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_1_161202_0: @(
while_lstm_cell_1_161204_0:@,
while_lstm_cell_1_161206_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_1_161202: @&
while_lstm_cell_1_161204:@*
while_lstm_cell_1_161206:@??)while/lstm_cell_1/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype0?
)while/lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_1_161202_0while_lstm_cell_1_161204_0while_lstm_cell_1_161206_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_161119?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity2while/lstm_cell_1/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:??????????
while/Identity_5Identity2while/lstm_cell_1/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:?????????x

while/NoOpNoOp*^while/lstm_cell_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_1_161202while_lstm_cell_1_161202_0"6
while_lstm_cell_1_161204while_lstm_cell_1_161204_0"6
while_lstm_cell_1_161206while_lstm_cell_1_161206_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????:?????????: : : : : 2V
)while/lstm_cell_1/StatefulPartitionedCall)while/lstm_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?

?
D__inference_dense_19_layer_call_and_return_conditional_losses_164260

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
??
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_162904

inputs6
#embedding_9_embedding_lookup_162509:	?' B
0lstm_1_lstm_cell_1_split_readvariableop_resource: @@
2lstm_1_lstm_cell_1_split_1_readvariableop_resource:@<
*lstm_1_lstm_cell_1_readvariableop_resource:@9
'dense_18_matmul_readvariableop_resource:
6
(dense_18_biasadd_readvariableop_resource:
9
'dense_19_matmul_readvariableop_resource:
6
(dense_19_biasadd_readvariableop_resource:
identity??dense_18/BiasAdd/ReadVariableOp?dense_18/MatMul/ReadVariableOp?dense_19/BiasAdd/ReadVariableOp?dense_19/MatMul/ReadVariableOp?embedding_9/embedding_lookup?!lstm_1/lstm_cell_1/ReadVariableOp?#lstm_1/lstm_cell_1/ReadVariableOp_1?#lstm_1/lstm_cell_1/ReadVariableOp_2?#lstm_1/lstm_cell_1/ReadVariableOp_3?'lstm_1/lstm_cell_1/split/ReadVariableOp?)lstm_1/lstm_cell_1/split_1/ReadVariableOp?lstm_1/whilej
embedding_9/CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:???????????????????
embedding_9/embedding_lookupResourceGather#embedding_9_embedding_lookup_162509embedding_9/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding_9/embedding_lookup/162509*4
_output_shapes"
 :?????????????????? *
dtype0?
%embedding_9/embedding_lookup/IdentityIdentity%embedding_9/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_9/embedding_lookup/162509*4
_output_shapes"
 :?????????????????? ?
'embedding_9/embedding_lookup/Identity_1Identity.embedding_9/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :?????????????????? l
lstm_1/ShapeShape0embedding_9/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:d
lstm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_1/strided_sliceStridedSlicelstm_1/Shape:output:0#lstm_1/strided_slice/stack:output:0%lstm_1/strided_slice/stack_1:output:0%lstm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
lstm_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :?
lstm_1/zeros/packedPacklstm_1/strided_slice:output:0lstm_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_1/zerosFilllstm_1/zeros/packed:output:0lstm_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????Y
lstm_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :?
lstm_1/zeros_1/packedPacklstm_1/strided_slice:output:0 lstm_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_1/zeros_1Filllstm_1/zeros_1/packed:output:0lstm_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????j
lstm_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_1/transpose	Transpose0embedding_9/embedding_lookup/Identity_1:output:0lstm_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? R
lstm_1/Shape_1Shapelstm_1/transpose:y:0*
T0*
_output_shapes
:f
lstm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_1/strided_slice_1StridedSlicelstm_1/Shape_1:output:0%lstm_1/strided_slice_1/stack:output:0'lstm_1/strided_slice_1/stack_1:output:0'lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
lstm_1/TensorArrayV2TensorListReserve+lstm_1/TensorArrayV2/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
.lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_1/transpose:y:0Elstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???f
lstm_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_1/strided_slice_2StridedSlicelstm_1/transpose:y:0%lstm_1/strided_slice_2/stack:output:0'lstm_1/strided_slice_2/stack_1:output:0'lstm_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_maskq
"lstm_1/lstm_cell_1/ones_like/ShapeShapelstm_1/strided_slice_2:output:0*
T0*
_output_shapes
:g
"lstm_1/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_1/lstm_cell_1/ones_likeFill+lstm_1/lstm_cell_1/ones_like/Shape:output:0+lstm_1/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? e
 lstm_1/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_1/lstm_cell_1/dropout/MulMul%lstm_1/lstm_cell_1/ones_like:output:0)lstm_1/lstm_cell_1/dropout/Const:output:0*
T0*'
_output_shapes
:????????? u
 lstm_1/lstm_cell_1/dropout/ShapeShape%lstm_1/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
7lstm_1/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform)lstm_1/lstm_cell_1/dropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???n
)lstm_1/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
'lstm_1/lstm_cell_1/dropout/GreaterEqualGreaterEqual@lstm_1/lstm_cell_1/dropout/random_uniform/RandomUniform:output:02lstm_1/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
lstm_1/lstm_cell_1/dropout/CastCast+lstm_1/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
 lstm_1/lstm_cell_1/dropout/Mul_1Mul"lstm_1/lstm_cell_1/dropout/Mul:z:0#lstm_1/lstm_cell_1/dropout/Cast:y:0*
T0*'
_output_shapes
:????????? g
"lstm_1/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
 lstm_1/lstm_cell_1/dropout_1/MulMul%lstm_1/lstm_cell_1/ones_like:output:0+lstm_1/lstm_cell_1/dropout_1/Const:output:0*
T0*'
_output_shapes
:????????? w
"lstm_1/lstm_cell_1/dropout_1/ShapeShape%lstm_1/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
9lstm_1/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform+lstm_1/lstm_cell_1/dropout_1/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???p
+lstm_1/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
)lstm_1/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualBlstm_1/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:04lstm_1/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
!lstm_1/lstm_cell_1/dropout_1/CastCast-lstm_1/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
"lstm_1/lstm_cell_1/dropout_1/Mul_1Mul$lstm_1/lstm_cell_1/dropout_1/Mul:z:0%lstm_1/lstm_cell_1/dropout_1/Cast:y:0*
T0*'
_output_shapes
:????????? g
"lstm_1/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
 lstm_1/lstm_cell_1/dropout_2/MulMul%lstm_1/lstm_cell_1/ones_like:output:0+lstm_1/lstm_cell_1/dropout_2/Const:output:0*
T0*'
_output_shapes
:????????? w
"lstm_1/lstm_cell_1/dropout_2/ShapeShape%lstm_1/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
9lstm_1/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform+lstm_1/lstm_cell_1/dropout_2/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2??>p
+lstm_1/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
)lstm_1/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualBlstm_1/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:04lstm_1/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
!lstm_1/lstm_cell_1/dropout_2/CastCast-lstm_1/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
"lstm_1/lstm_cell_1/dropout_2/Mul_1Mul$lstm_1/lstm_cell_1/dropout_2/Mul:z:0%lstm_1/lstm_cell_1/dropout_2/Cast:y:0*
T0*'
_output_shapes
:????????? g
"lstm_1/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
 lstm_1/lstm_cell_1/dropout_3/MulMul%lstm_1/lstm_cell_1/ones_like:output:0+lstm_1/lstm_cell_1/dropout_3/Const:output:0*
T0*'
_output_shapes
:????????? w
"lstm_1/lstm_cell_1/dropout_3/ShapeShape%lstm_1/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
9lstm_1/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform+lstm_1/lstm_cell_1/dropout_3/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2?ʈp
+lstm_1/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
)lstm_1/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualBlstm_1/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:04lstm_1/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
!lstm_1/lstm_cell_1/dropout_3/CastCast-lstm_1/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
"lstm_1/lstm_cell_1/dropout_3/Mul_1Mul$lstm_1/lstm_cell_1/dropout_3/Mul:z:0%lstm_1/lstm_cell_1/dropout_3/Cast:y:0*
T0*'
_output_shapes
:????????? i
$lstm_1/lstm_cell_1/ones_like_1/ShapeShapelstm_1/zeros:output:0*
T0*
_output_shapes
:i
$lstm_1/lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_1/lstm_cell_1/ones_like_1Fill-lstm_1/lstm_cell_1/ones_like_1/Shape:output:0-lstm_1/lstm_cell_1/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????g
"lstm_1/lstm_cell_1/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
 lstm_1/lstm_cell_1/dropout_4/MulMul'lstm_1/lstm_cell_1/ones_like_1:output:0+lstm_1/lstm_cell_1/dropout_4/Const:output:0*
T0*'
_output_shapes
:?????????y
"lstm_1/lstm_cell_1/dropout_4/ShapeShape'lstm_1/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
9lstm_1/lstm_cell_1/dropout_4/random_uniform/RandomUniformRandomUniform+lstm_1/lstm_cell_1/dropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???p
+lstm_1/lstm_cell_1/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
)lstm_1/lstm_cell_1/dropout_4/GreaterEqualGreaterEqualBlstm_1/lstm_cell_1/dropout_4/random_uniform/RandomUniform:output:04lstm_1/lstm_cell_1/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
!lstm_1/lstm_cell_1/dropout_4/CastCast-lstm_1/lstm_cell_1/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
"lstm_1/lstm_cell_1/dropout_4/Mul_1Mul$lstm_1/lstm_cell_1/dropout_4/Mul:z:0%lstm_1/lstm_cell_1/dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????g
"lstm_1/lstm_cell_1/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
 lstm_1/lstm_cell_1/dropout_5/MulMul'lstm_1/lstm_cell_1/ones_like_1:output:0+lstm_1/lstm_cell_1/dropout_5/Const:output:0*
T0*'
_output_shapes
:?????????y
"lstm_1/lstm_cell_1/dropout_5/ShapeShape'lstm_1/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
9lstm_1/lstm_cell_1/dropout_5/random_uniform/RandomUniformRandomUniform+lstm_1/lstm_cell_1/dropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???p
+lstm_1/lstm_cell_1/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
)lstm_1/lstm_cell_1/dropout_5/GreaterEqualGreaterEqualBlstm_1/lstm_cell_1/dropout_5/random_uniform/RandomUniform:output:04lstm_1/lstm_cell_1/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
!lstm_1/lstm_cell_1/dropout_5/CastCast-lstm_1/lstm_cell_1/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
"lstm_1/lstm_cell_1/dropout_5/Mul_1Mul$lstm_1/lstm_cell_1/dropout_5/Mul:z:0%lstm_1/lstm_cell_1/dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????g
"lstm_1/lstm_cell_1/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
 lstm_1/lstm_cell_1/dropout_6/MulMul'lstm_1/lstm_cell_1/ones_like_1:output:0+lstm_1/lstm_cell_1/dropout_6/Const:output:0*
T0*'
_output_shapes
:?????????y
"lstm_1/lstm_cell_1/dropout_6/ShapeShape'lstm_1/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
9lstm_1/lstm_cell_1/dropout_6/random_uniform/RandomUniformRandomUniform+lstm_1/lstm_cell_1/dropout_6/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2͋?p
+lstm_1/lstm_cell_1/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
)lstm_1/lstm_cell_1/dropout_6/GreaterEqualGreaterEqualBlstm_1/lstm_cell_1/dropout_6/random_uniform/RandomUniform:output:04lstm_1/lstm_cell_1/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
!lstm_1/lstm_cell_1/dropout_6/CastCast-lstm_1/lstm_cell_1/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
"lstm_1/lstm_cell_1/dropout_6/Mul_1Mul$lstm_1/lstm_cell_1/dropout_6/Mul:z:0%lstm_1/lstm_cell_1/dropout_6/Cast:y:0*
T0*'
_output_shapes
:?????????g
"lstm_1/lstm_cell_1/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
 lstm_1/lstm_cell_1/dropout_7/MulMul'lstm_1/lstm_cell_1/ones_like_1:output:0+lstm_1/lstm_cell_1/dropout_7/Const:output:0*
T0*'
_output_shapes
:?????????y
"lstm_1/lstm_cell_1/dropout_7/ShapeShape'lstm_1/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
9lstm_1/lstm_cell_1/dropout_7/random_uniform/RandomUniformRandomUniform+lstm_1/lstm_cell_1/dropout_7/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???p
+lstm_1/lstm_cell_1/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
)lstm_1/lstm_cell_1/dropout_7/GreaterEqualGreaterEqualBlstm_1/lstm_cell_1/dropout_7/random_uniform/RandomUniform:output:04lstm_1/lstm_cell_1/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
!lstm_1/lstm_cell_1/dropout_7/CastCast-lstm_1/lstm_cell_1/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
"lstm_1/lstm_cell_1/dropout_7/Mul_1Mul$lstm_1/lstm_cell_1/dropout_7/Mul:z:0%lstm_1/lstm_cell_1/dropout_7/Cast:y:0*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/mulMullstm_1/strided_slice_2:output:0$lstm_1/lstm_cell_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
lstm_1/lstm_cell_1/mul_1Mullstm_1/strided_slice_2:output:0&lstm_1/lstm_cell_1/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
lstm_1/lstm_cell_1/mul_2Mullstm_1/strided_slice_2:output:0&lstm_1/lstm_cell_1/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
lstm_1/lstm_cell_1/mul_3Mullstm_1/strided_slice_2:output:0&lstm_1/lstm_cell_1/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:????????? d
"lstm_1/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
'lstm_1/lstm_cell_1/split/ReadVariableOpReadVariableOp0lstm_1_lstm_cell_1_split_readvariableop_resource*
_output_shapes

: @*
dtype0?
lstm_1/lstm_cell_1/splitSplit+lstm_1/lstm_cell_1/split/split_dim:output:0/lstm_1/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split?
lstm_1/lstm_cell_1/MatMulMatMullstm_1/lstm_cell_1/mul:z:0!lstm_1/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/MatMul_1MatMullstm_1/lstm_cell_1/mul_1:z:0!lstm_1/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/MatMul_2MatMullstm_1/lstm_cell_1/mul_2:z:0!lstm_1/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/MatMul_3MatMullstm_1/lstm_cell_1/mul_3:z:0!lstm_1/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????f
$lstm_1/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
)lstm_1/lstm_cell_1/split_1/ReadVariableOpReadVariableOp2lstm_1_lstm_cell_1_split_1_readvariableop_resource*
_output_shapes
:@*
dtype0?
lstm_1/lstm_cell_1/split_1Split-lstm_1/lstm_cell_1/split_1/split_dim:output:01lstm_1/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
lstm_1/lstm_cell_1/BiasAddBiasAdd#lstm_1/lstm_cell_1/MatMul:product:0#lstm_1/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/BiasAdd_1BiasAdd%lstm_1/lstm_cell_1/MatMul_1:product:0#lstm_1/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/BiasAdd_2BiasAdd%lstm_1/lstm_cell_1/MatMul_2:product:0#lstm_1/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/BiasAdd_3BiasAdd%lstm_1/lstm_cell_1/MatMul_3:product:0#lstm_1/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/mul_4Mullstm_1/zeros:output:0&lstm_1/lstm_cell_1/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/mul_5Mullstm_1/zeros:output:0&lstm_1/lstm_cell_1/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/mul_6Mullstm_1/zeros:output:0&lstm_1/lstm_cell_1/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/mul_7Mullstm_1/zeros:output:0&lstm_1/lstm_cell_1/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:??????????
!lstm_1/lstm_cell_1/ReadVariableOpReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0w
&lstm_1/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(lstm_1/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(lstm_1/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
 lstm_1/lstm_cell_1/strided_sliceStridedSlice)lstm_1/lstm_cell_1/ReadVariableOp:value:0/lstm_1/lstm_cell_1/strided_slice/stack:output:01lstm_1/lstm_cell_1/strided_slice/stack_1:output:01lstm_1/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_1/lstm_cell_1/MatMul_4MatMullstm_1/lstm_cell_1/mul_4:z:0)lstm_1/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/addAddV2#lstm_1/lstm_cell_1/BiasAdd:output:0%lstm_1/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????s
lstm_1/lstm_cell_1/SigmoidSigmoidlstm_1/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
#lstm_1/lstm_cell_1/ReadVariableOp_1ReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0y
(lstm_1/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_1/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_1/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_1/lstm_cell_1/strided_slice_1StridedSlice+lstm_1/lstm_cell_1/ReadVariableOp_1:value:01lstm_1/lstm_cell_1/strided_slice_1/stack:output:03lstm_1/lstm_cell_1/strided_slice_1/stack_1:output:03lstm_1/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_1/lstm_cell_1/MatMul_5MatMullstm_1/lstm_cell_1/mul_5:z:0+lstm_1/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/add_1AddV2%lstm_1/lstm_cell_1/BiasAdd_1:output:0%lstm_1/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????w
lstm_1/lstm_cell_1/Sigmoid_1Sigmoidlstm_1/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/mul_8Mul lstm_1/lstm_cell_1/Sigmoid_1:y:0lstm_1/zeros_1:output:0*
T0*'
_output_shapes
:??????????
#lstm_1/lstm_cell_1/ReadVariableOp_2ReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0y
(lstm_1/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_1/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   {
*lstm_1/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_1/lstm_cell_1/strided_slice_2StridedSlice+lstm_1/lstm_cell_1/ReadVariableOp_2:value:01lstm_1/lstm_cell_1/strided_slice_2/stack:output:03lstm_1/lstm_cell_1/strided_slice_2/stack_1:output:03lstm_1/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_1/lstm_cell_1/MatMul_6MatMullstm_1/lstm_cell_1/mul_6:z:0+lstm_1/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/add_2AddV2%lstm_1/lstm_cell_1/BiasAdd_2:output:0%lstm_1/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????o
lstm_1/lstm_cell_1/TanhTanhlstm_1/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/mul_9Mullstm_1/lstm_cell_1/Sigmoid:y:0lstm_1/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/add_3AddV2lstm_1/lstm_cell_1/mul_8:z:0lstm_1/lstm_cell_1/mul_9:z:0*
T0*'
_output_shapes
:??????????
#lstm_1/lstm_cell_1/ReadVariableOp_3ReadVariableOp*lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0y
(lstm_1/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   {
*lstm_1/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_1/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_1/lstm_cell_1/strided_slice_3StridedSlice+lstm_1/lstm_cell_1/ReadVariableOp_3:value:01lstm_1/lstm_cell_1/strided_slice_3/stack:output:03lstm_1/lstm_cell_1/strided_slice_3/stack_1:output:03lstm_1/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_1/lstm_cell_1/MatMul_7MatMullstm_1/lstm_cell_1/mul_7:z:0+lstm_1/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/add_4AddV2%lstm_1/lstm_cell_1/BiasAdd_3:output:0%lstm_1/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????w
lstm_1/lstm_cell_1/Sigmoid_2Sigmoidlstm_1/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????q
lstm_1/lstm_cell_1/Tanh_1Tanhlstm_1/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:??????????
lstm_1/lstm_cell_1/mul_10Mul lstm_1/lstm_cell_1/Sigmoid_2:y:0lstm_1/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????u
$lstm_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
lstm_1/TensorArrayV2_1TensorListReserve-lstm_1/TensorArrayV2_1/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???M
lstm_1/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????[
lstm_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
lstm_1/whileWhile"lstm_1/while/loop_counter:output:0(lstm_1/while/maximum_iterations:output:0lstm_1/time:output:0lstm_1/TensorArrayV2_1:handle:0lstm_1/zeros:output:0lstm_1/zeros_1:output:0lstm_1/strided_slice_1:output:0>lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:00lstm_1_lstm_cell_1_split_readvariableop_resource2lstm_1_lstm_cell_1_split_1_readvariableop_resource*lstm_1_lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_1_while_body_162684*$
condR
lstm_1_while_cond_162683*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations ?
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)lstm_1/TensorArrayV2Stack/TensorListStackTensorListStacklstm_1/while:output:3@lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0o
lstm_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????h
lstm_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_1/strided_slice_3StridedSlice2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_1/strided_slice_3/stack:output:0'lstm_1/strided_slice_3/stack_1:output:0'lstm_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskl
lstm_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_1/transpose_1	Transpose2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_1/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????b
lstm_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ?
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
dense_18/MatMulMatMullstm_1/strided_slice_3:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
b
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
\
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_9/dropout/MulMuldense_18/Relu:activations:0 dropout_9/dropout/Const:output:0*
T0*'
_output_shapes
:?????????
b
dropout_9/dropout/ShapeShapedense_18/Relu:activations:0*
T0*
_output_shapes
:?
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????
*
dtype0e
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????
?
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
?
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????
?
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
dense_19/MatMulMatMuldropout_9/dropout/Mul_1:z:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_19/SigmoidSigmoiddense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
IdentityIdentitydense_19/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp^embedding_9/embedding_lookup"^lstm_1/lstm_cell_1/ReadVariableOp$^lstm_1/lstm_cell_1/ReadVariableOp_1$^lstm_1/lstm_cell_1/ReadVariableOp_2$^lstm_1/lstm_cell_1/ReadVariableOp_3(^lstm_1/lstm_cell_1/split/ReadVariableOp*^lstm_1/lstm_cell_1/split_1/ReadVariableOp^lstm_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????????????: : : : : : : : 2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2<
embedding_9/embedding_lookupembedding_9/embedding_lookup2F
!lstm_1/lstm_cell_1/ReadVariableOp!lstm_1/lstm_cell_1/ReadVariableOp2J
#lstm_1/lstm_cell_1/ReadVariableOp_1#lstm_1/lstm_cell_1/ReadVariableOp_12J
#lstm_1/lstm_cell_1/ReadVariableOp_2#lstm_1/lstm_cell_1/ReadVariableOp_22J
#lstm_1/lstm_cell_1/ReadVariableOp_3#lstm_1/lstm_cell_1/ReadVariableOp_32R
'lstm_1/lstm_cell_1/split/ReadVariableOp'lstm_1/lstm_cell_1/split/ReadVariableOp2V
)lstm_1/lstm_cell_1/split_1/ReadVariableOp)lstm_1/lstm_cell_1/split_1/ReadVariableOp2
lstm_1/whilelstm_1/while:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
)__inference_dense_19_layer_call_fn_164249

inputs
unknown:

	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_161562o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?

?
-__inference_sequential_9_layer_call_fn_162120
embedding_9_input
unknown:	?' 
	unknown_0: @
	unknown_1:@
	unknown_2:@
	unknown_3:

	unknown_4:

	unknown_5:

	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallembedding_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_162080o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
0
_output_shapes
:??????????????????
+
_user_specified_nameembedding_9_input
?
F
*__inference_dropout_9_layer_call_fn_164218

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_161549`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?

?
D__inference_dense_18_layer_call_and_return_conditional_losses_161538

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?7
?
B__inference_lstm_1_layer_call_and_return_conditional_losses_161247

inputs$
lstm_cell_1_161165: @ 
lstm_cell_1_161167:@$
lstm_cell_1_161169:@
identity??#lstm_cell_1/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask?
#lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_1_161165lstm_cell_1_161167lstm_cell_1_161169*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_161119n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_1_161165lstm_cell_1_161167lstm_cell_1_161169*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_161178*
condR
while_cond_161177*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????t
NoOpNoOp$^lstm_cell_1/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????????????? : : : 2J
#lstm_cell_1/StatefulPartitionedCall#lstm_cell_1/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?	
?
lstm_1_while_cond_162683*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3,
(lstm_1_while_less_lstm_1_strided_slice_1B
>lstm_1_while_lstm_1_while_cond_162683___redundant_placeholder0B
>lstm_1_while_lstm_1_while_cond_162683___redundant_placeholder1B
>lstm_1_while_lstm_1_while_cond_162683___redundant_placeholder2B
>lstm_1_while_lstm_1_while_cond_162683___redundant_placeholder3
lstm_1_while_identity
~
lstm_1/while/LessLesslstm_1_while_placeholder(lstm_1_while_less_lstm_1_strided_slice_1*
T0*
_output_shapes
: Y
lstm_1/while/IdentityIdentitylstm_1/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_1_while_identitylstm_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?

?
-__inference_sequential_9_layer_call_fn_161588
embedding_9_input
unknown:	?' 
	unknown_0: @
	unknown_1:@
	unknown_2:@
	unknown_3:

	unknown_4:

	unknown_5:

	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallembedding_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_161569o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
0
_output_shapes
:??????????????????
+
_user_specified_nameembedding_9_input
??
?
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_161119

inputs

states
states_1/
split_readvariableop_resource: @-
split_1_readvariableop_resource:@)
readvariableop_resource:@
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:????????? R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??p
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:????????? O
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? T
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??t
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:????????? Q
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? s
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? o
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:????????? T
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??t
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:????????? Q
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2??w]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? s
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? o
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:????????? T
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??t
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:????????? Q
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2??y]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? s
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? o
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:????????? G
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??}
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????T
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??v
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*'
_output_shapes
:?????????S
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:?
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???]
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????s
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????o
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????T
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??v
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*'
_output_shapes
:?????????S
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:?
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2븬]
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????s
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????o
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????T
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??v
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*'
_output_shapes
:?????????S
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:?
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???]
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????s
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????o
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*'
_output_shapes
:?????????T
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??v
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*'
_output_shapes
:?????????S
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:?
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???]
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????s
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????o
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*'
_output_shapes
:?????????W
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? [
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:????????? [
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*'
_output_shapes
:????????? [
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:????????? Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :r
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

: @*
dtype0?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split[
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:?????????_
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:?????????_
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:?????????_
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:?????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:@*
dtype0?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????[
mul_4Mulstatesdropout_4/Mul_1:z:0*
T0*'
_output_shapes
:?????????[
mul_5Mulstatesdropout_5/Mul_1:z:0*
T0*'
_output_shapes
:?????????[
mul_6Mulstatesdropout_6/Mul_1:z:0*
T0*'
_output_shapes
:?????????[
mul_7Mulstatesdropout_7/Mul_1:z:0*
T0*'
_output_shapes
:?????????f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskg
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:?????????d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:?????????h
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????W
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????h
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????U
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????V
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:?????????h
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????h
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????Q
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:?????????K
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:?????????Z
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????Y
IdentityIdentity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:?????????[

Identity_1Identity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:?????????Z

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:????????? :?????????:?????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_namestates:OK
'
_output_shapes
:?????????
 
_user_specified_namestates
?
?
,__inference_embedding_9_layer_call_fn_162911

inputs
unknown:	?' 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_9_layer_call_and_return_conditional_losses_161273|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?

?
D__inference_dense_19_layer_call_and_return_conditional_losses_161562

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
??
?	
!__inference__wrapped_model_160742
embedding_9_inputC
0sequential_9_embedding_9_embedding_lookup_160482:	?' O
=sequential_9_lstm_1_lstm_cell_1_split_readvariableop_resource: @M
?sequential_9_lstm_1_lstm_cell_1_split_1_readvariableop_resource:@I
7sequential_9_lstm_1_lstm_cell_1_readvariableop_resource:@F
4sequential_9_dense_18_matmul_readvariableop_resource:
C
5sequential_9_dense_18_biasadd_readvariableop_resource:
F
4sequential_9_dense_19_matmul_readvariableop_resource:
C
5sequential_9_dense_19_biasadd_readvariableop_resource:
identity??,sequential_9/dense_18/BiasAdd/ReadVariableOp?+sequential_9/dense_18/MatMul/ReadVariableOp?,sequential_9/dense_19/BiasAdd/ReadVariableOp?+sequential_9/dense_19/MatMul/ReadVariableOp?)sequential_9/embedding_9/embedding_lookup?.sequential_9/lstm_1/lstm_cell_1/ReadVariableOp?0sequential_9/lstm_1/lstm_cell_1/ReadVariableOp_1?0sequential_9/lstm_1/lstm_cell_1/ReadVariableOp_2?0sequential_9/lstm_1/lstm_cell_1/ReadVariableOp_3?4sequential_9/lstm_1/lstm_cell_1/split/ReadVariableOp?6sequential_9/lstm_1/lstm_cell_1/split_1/ReadVariableOp?sequential_9/lstm_1/while?
sequential_9/embedding_9/CastCastembedding_9_input*

DstT0*

SrcT0*0
_output_shapes
:???????????????????
)sequential_9/embedding_9/embedding_lookupResourceGather0sequential_9_embedding_9_embedding_lookup_160482!sequential_9/embedding_9/Cast:y:0*
Tindices0*C
_class9
75loc:@sequential_9/embedding_9/embedding_lookup/160482*4
_output_shapes"
 :?????????????????? *
dtype0?
2sequential_9/embedding_9/embedding_lookup/IdentityIdentity2sequential_9/embedding_9/embedding_lookup:output:0*
T0*C
_class9
75loc:@sequential_9/embedding_9/embedding_lookup/160482*4
_output_shapes"
 :?????????????????? ?
4sequential_9/embedding_9/embedding_lookup/Identity_1Identity;sequential_9/embedding_9/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :?????????????????? ?
sequential_9/lstm_1/ShapeShape=sequential_9/embedding_9/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:q
'sequential_9/lstm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential_9/lstm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential_9/lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!sequential_9/lstm_1/strided_sliceStridedSlice"sequential_9/lstm_1/Shape:output:00sequential_9/lstm_1/strided_slice/stack:output:02sequential_9/lstm_1/strided_slice/stack_1:output:02sequential_9/lstm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sequential_9/lstm_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :?
 sequential_9/lstm_1/zeros/packedPack*sequential_9/lstm_1/strided_slice:output:0+sequential_9/lstm_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_9/lstm_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential_9/lstm_1/zerosFill)sequential_9/lstm_1/zeros/packed:output:0(sequential_9/lstm_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????f
$sequential_9/lstm_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :?
"sequential_9/lstm_1/zeros_1/packedPack*sequential_9/lstm_1/strided_slice:output:0-sequential_9/lstm_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_9/lstm_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential_9/lstm_1/zeros_1Fill+sequential_9/lstm_1/zeros_1/packed:output:0*sequential_9/lstm_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????w
"sequential_9/lstm_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
sequential_9/lstm_1/transpose	Transpose=sequential_9/embedding_9/embedding_lookup/Identity_1:output:0+sequential_9/lstm_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? l
sequential_9/lstm_1/Shape_1Shape!sequential_9/lstm_1/transpose:y:0*
T0*
_output_shapes
:s
)sequential_9/lstm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_9/lstm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_9/lstm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#sequential_9/lstm_1/strided_slice_1StridedSlice$sequential_9/lstm_1/Shape_1:output:02sequential_9/lstm_1/strided_slice_1/stack:output:04sequential_9/lstm_1/strided_slice_1/stack_1:output:04sequential_9/lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/sequential_9/lstm_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
!sequential_9/lstm_1/TensorArrayV2TensorListReserve8sequential_9/lstm_1/TensorArrayV2/element_shape:output:0,sequential_9/lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Isequential_9/lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
;sequential_9/lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_9/lstm_1/transpose:y:0Rsequential_9/lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???s
)sequential_9/lstm_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_9/lstm_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_9/lstm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#sequential_9/lstm_1/strided_slice_2StridedSlice!sequential_9/lstm_1/transpose:y:02sequential_9/lstm_1/strided_slice_2/stack:output:04sequential_9/lstm_1/strided_slice_2/stack_1:output:04sequential_9/lstm_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask?
/sequential_9/lstm_1/lstm_cell_1/ones_like/ShapeShape,sequential_9/lstm_1/strided_slice_2:output:0*
T0*
_output_shapes
:t
/sequential_9/lstm_1/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)sequential_9/lstm_1/lstm_cell_1/ones_likeFill8sequential_9/lstm_1/lstm_cell_1/ones_like/Shape:output:08sequential_9/lstm_1/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? ?
1sequential_9/lstm_1/lstm_cell_1/ones_like_1/ShapeShape"sequential_9/lstm_1/zeros:output:0*
T0*
_output_shapes
:v
1sequential_9/lstm_1/lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
+sequential_9/lstm_1/lstm_cell_1/ones_like_1Fill:sequential_9/lstm_1/lstm_cell_1/ones_like_1/Shape:output:0:sequential_9/lstm_1/lstm_cell_1/ones_like_1/Const:output:0*
T0*'
_output_shapes
:??????????
#sequential_9/lstm_1/lstm_cell_1/mulMul,sequential_9/lstm_1/strided_slice_2:output:02sequential_9/lstm_1/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
%sequential_9/lstm_1/lstm_cell_1/mul_1Mul,sequential_9/lstm_1/strided_slice_2:output:02sequential_9/lstm_1/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
%sequential_9/lstm_1/lstm_cell_1/mul_2Mul,sequential_9/lstm_1/strided_slice_2:output:02sequential_9/lstm_1/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
%sequential_9/lstm_1/lstm_cell_1/mul_3Mul,sequential_9/lstm_1/strided_slice_2:output:02sequential_9/lstm_1/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? q
/sequential_9/lstm_1/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
4sequential_9/lstm_1/lstm_cell_1/split/ReadVariableOpReadVariableOp=sequential_9_lstm_1_lstm_cell_1_split_readvariableop_resource*
_output_shapes

: @*
dtype0?
%sequential_9/lstm_1/lstm_cell_1/splitSplit8sequential_9/lstm_1/lstm_cell_1/split/split_dim:output:0<sequential_9/lstm_1/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split?
&sequential_9/lstm_1/lstm_cell_1/MatMulMatMul'sequential_9/lstm_1/lstm_cell_1/mul:z:0.sequential_9/lstm_1/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:??????????
(sequential_9/lstm_1/lstm_cell_1/MatMul_1MatMul)sequential_9/lstm_1/lstm_cell_1/mul_1:z:0.sequential_9/lstm_1/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:??????????
(sequential_9/lstm_1/lstm_cell_1/MatMul_2MatMul)sequential_9/lstm_1/lstm_cell_1/mul_2:z:0.sequential_9/lstm_1/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:??????????
(sequential_9/lstm_1/lstm_cell_1/MatMul_3MatMul)sequential_9/lstm_1/lstm_cell_1/mul_3:z:0.sequential_9/lstm_1/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????s
1sequential_9/lstm_1/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
6sequential_9/lstm_1/lstm_cell_1/split_1/ReadVariableOpReadVariableOp?sequential_9_lstm_1_lstm_cell_1_split_1_readvariableop_resource*
_output_shapes
:@*
dtype0?
'sequential_9/lstm_1/lstm_cell_1/split_1Split:sequential_9/lstm_1/lstm_cell_1/split_1/split_dim:output:0>sequential_9/lstm_1/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
'sequential_9/lstm_1/lstm_cell_1/BiasAddBiasAdd0sequential_9/lstm_1/lstm_cell_1/MatMul:product:00sequential_9/lstm_1/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:??????????
)sequential_9/lstm_1/lstm_cell_1/BiasAdd_1BiasAdd2sequential_9/lstm_1/lstm_cell_1/MatMul_1:product:00sequential_9/lstm_1/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:??????????
)sequential_9/lstm_1/lstm_cell_1/BiasAdd_2BiasAdd2sequential_9/lstm_1/lstm_cell_1/MatMul_2:product:00sequential_9/lstm_1/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
)sequential_9/lstm_1/lstm_cell_1/BiasAdd_3BiasAdd2sequential_9/lstm_1/lstm_cell_1/MatMul_3:product:00sequential_9/lstm_1/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:??????????
%sequential_9/lstm_1/lstm_cell_1/mul_4Mul"sequential_9/lstm_1/zeros:output:04sequential_9/lstm_1/lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
%sequential_9/lstm_1/lstm_cell_1/mul_5Mul"sequential_9/lstm_1/zeros:output:04sequential_9/lstm_1/lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
%sequential_9/lstm_1/lstm_cell_1/mul_6Mul"sequential_9/lstm_1/zeros:output:04sequential_9/lstm_1/lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
%sequential_9/lstm_1/lstm_cell_1/mul_7Mul"sequential_9/lstm_1/zeros:output:04sequential_9/lstm_1/lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
.sequential_9/lstm_1/lstm_cell_1/ReadVariableOpReadVariableOp7sequential_9_lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0?
3sequential_9/lstm_1/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
5sequential_9/lstm_1/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
5sequential_9/lstm_1/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
-sequential_9/lstm_1/lstm_cell_1/strided_sliceStridedSlice6sequential_9/lstm_1/lstm_cell_1/ReadVariableOp:value:0<sequential_9/lstm_1/lstm_cell_1/strided_slice/stack:output:0>sequential_9/lstm_1/lstm_cell_1/strided_slice/stack_1:output:0>sequential_9/lstm_1/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
(sequential_9/lstm_1/lstm_cell_1/MatMul_4MatMul)sequential_9/lstm_1/lstm_cell_1/mul_4:z:06sequential_9/lstm_1/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:??????????
#sequential_9/lstm_1/lstm_cell_1/addAddV20sequential_9/lstm_1/lstm_cell_1/BiasAdd:output:02sequential_9/lstm_1/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:??????????
'sequential_9/lstm_1/lstm_cell_1/SigmoidSigmoid'sequential_9/lstm_1/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
0sequential_9/lstm_1/lstm_cell_1/ReadVariableOp_1ReadVariableOp7sequential_9_lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0?
5sequential_9/lstm_1/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       ?
7sequential_9/lstm_1/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
7sequential_9/lstm_1/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
/sequential_9/lstm_1/lstm_cell_1/strided_slice_1StridedSlice8sequential_9/lstm_1/lstm_cell_1/ReadVariableOp_1:value:0>sequential_9/lstm_1/lstm_cell_1/strided_slice_1/stack:output:0@sequential_9/lstm_1/lstm_cell_1/strided_slice_1/stack_1:output:0@sequential_9/lstm_1/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
(sequential_9/lstm_1/lstm_cell_1/MatMul_5MatMul)sequential_9/lstm_1/lstm_cell_1/mul_5:z:08sequential_9/lstm_1/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
%sequential_9/lstm_1/lstm_cell_1/add_1AddV22sequential_9/lstm_1/lstm_cell_1/BiasAdd_1:output:02sequential_9/lstm_1/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:??????????
)sequential_9/lstm_1/lstm_cell_1/Sigmoid_1Sigmoid)sequential_9/lstm_1/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:??????????
%sequential_9/lstm_1/lstm_cell_1/mul_8Mul-sequential_9/lstm_1/lstm_cell_1/Sigmoid_1:y:0$sequential_9/lstm_1/zeros_1:output:0*
T0*'
_output_shapes
:??????????
0sequential_9/lstm_1/lstm_cell_1/ReadVariableOp_2ReadVariableOp7sequential_9_lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0?
5sequential_9/lstm_1/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
7sequential_9/lstm_1/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   ?
7sequential_9/lstm_1/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
/sequential_9/lstm_1/lstm_cell_1/strided_slice_2StridedSlice8sequential_9/lstm_1/lstm_cell_1/ReadVariableOp_2:value:0>sequential_9/lstm_1/lstm_cell_1/strided_slice_2/stack:output:0@sequential_9/lstm_1/lstm_cell_1/strided_slice_2/stack_1:output:0@sequential_9/lstm_1/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
(sequential_9/lstm_1/lstm_cell_1/MatMul_6MatMul)sequential_9/lstm_1/lstm_cell_1/mul_6:z:08sequential_9/lstm_1/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
%sequential_9/lstm_1/lstm_cell_1/add_2AddV22sequential_9/lstm_1/lstm_cell_1/BiasAdd_2:output:02sequential_9/lstm_1/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:??????????
$sequential_9/lstm_1/lstm_cell_1/TanhTanh)sequential_9/lstm_1/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:??????????
%sequential_9/lstm_1/lstm_cell_1/mul_9Mul+sequential_9/lstm_1/lstm_cell_1/Sigmoid:y:0(sequential_9/lstm_1/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:??????????
%sequential_9/lstm_1/lstm_cell_1/add_3AddV2)sequential_9/lstm_1/lstm_cell_1/mul_8:z:0)sequential_9/lstm_1/lstm_cell_1/mul_9:z:0*
T0*'
_output_shapes
:??????????
0sequential_9/lstm_1/lstm_cell_1/ReadVariableOp_3ReadVariableOp7sequential_9_lstm_1_lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0?
5sequential_9/lstm_1/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   ?
7sequential_9/lstm_1/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
7sequential_9/lstm_1/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
/sequential_9/lstm_1/lstm_cell_1/strided_slice_3StridedSlice8sequential_9/lstm_1/lstm_cell_1/ReadVariableOp_3:value:0>sequential_9/lstm_1/lstm_cell_1/strided_slice_3/stack:output:0@sequential_9/lstm_1/lstm_cell_1/strided_slice_3/stack_1:output:0@sequential_9/lstm_1/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
(sequential_9/lstm_1/lstm_cell_1/MatMul_7MatMul)sequential_9/lstm_1/lstm_cell_1/mul_7:z:08sequential_9/lstm_1/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
%sequential_9/lstm_1/lstm_cell_1/add_4AddV22sequential_9/lstm_1/lstm_cell_1/BiasAdd_3:output:02sequential_9/lstm_1/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:??????????
)sequential_9/lstm_1/lstm_cell_1/Sigmoid_2Sigmoid)sequential_9/lstm_1/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:??????????
&sequential_9/lstm_1/lstm_cell_1/Tanh_1Tanh)sequential_9/lstm_1/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:??????????
&sequential_9/lstm_1/lstm_cell_1/mul_10Mul-sequential_9/lstm_1/lstm_cell_1/Sigmoid_2:y:0*sequential_9/lstm_1/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:??????????
1sequential_9/lstm_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
#sequential_9/lstm_1/TensorArrayV2_1TensorListReserve:sequential_9/lstm_1/TensorArrayV2_1/element_shape:output:0,sequential_9/lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???Z
sequential_9/lstm_1/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,sequential_9/lstm_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????h
&sequential_9/lstm_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
sequential_9/lstm_1/whileWhile/sequential_9/lstm_1/while/loop_counter:output:05sequential_9/lstm_1/while/maximum_iterations:output:0!sequential_9/lstm_1/time:output:0,sequential_9/lstm_1/TensorArrayV2_1:handle:0"sequential_9/lstm_1/zeros:output:0$sequential_9/lstm_1/zeros_1:output:0,sequential_9/lstm_1/strided_slice_1:output:0Ksequential_9/lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0=sequential_9_lstm_1_lstm_cell_1_split_readvariableop_resource?sequential_9_lstm_1_lstm_cell_1_split_1_readvariableop_resource7sequential_9_lstm_1_lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *1
body)R'
%sequential_9_lstm_1_while_body_160593*1
cond)R'
%sequential_9_lstm_1_while_cond_160592*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations ?
Dsequential_9/lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
6sequential_9/lstm_1/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_9/lstm_1/while:output:3Msequential_9/lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0|
)sequential_9/lstm_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????u
+sequential_9/lstm_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+sequential_9/lstm_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#sequential_9/lstm_1/strided_slice_3StridedSlice?sequential_9/lstm_1/TensorArrayV2Stack/TensorListStack:tensor:02sequential_9/lstm_1/strided_slice_3/stack:output:04sequential_9/lstm_1/strided_slice_3/stack_1:output:04sequential_9/lstm_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_masky
$sequential_9/lstm_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
sequential_9/lstm_1/transpose_1	Transpose?sequential_9/lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_9/lstm_1/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????o
sequential_9/lstm_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ?
+sequential_9/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_18_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
sequential_9/dense_18/MatMulMatMul,sequential_9/lstm_1/strided_slice_3:output:03sequential_9/dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
,sequential_9/dense_18/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_18_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
sequential_9/dense_18/BiasAddBiasAdd&sequential_9/dense_18/MatMul:product:04sequential_9/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
|
sequential_9/dense_18/ReluRelu&sequential_9/dense_18/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
sequential_9/dropout_9/IdentityIdentity(sequential_9/dense_18/Relu:activations:0*
T0*'
_output_shapes
:?????????
?
+sequential_9/dense_19/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_19_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
sequential_9/dense_19/MatMulMatMul(sequential_9/dropout_9/Identity:output:03sequential_9/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
,sequential_9/dense_19/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_9/dense_19/BiasAddBiasAdd&sequential_9/dense_19/MatMul:product:04sequential_9/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
sequential_9/dense_19/SigmoidSigmoid&sequential_9/dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????p
IdentityIdentity!sequential_9/dense_19/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp-^sequential_9/dense_18/BiasAdd/ReadVariableOp,^sequential_9/dense_18/MatMul/ReadVariableOp-^sequential_9/dense_19/BiasAdd/ReadVariableOp,^sequential_9/dense_19/MatMul/ReadVariableOp*^sequential_9/embedding_9/embedding_lookup/^sequential_9/lstm_1/lstm_cell_1/ReadVariableOp1^sequential_9/lstm_1/lstm_cell_1/ReadVariableOp_11^sequential_9/lstm_1/lstm_cell_1/ReadVariableOp_21^sequential_9/lstm_1/lstm_cell_1/ReadVariableOp_35^sequential_9/lstm_1/lstm_cell_1/split/ReadVariableOp7^sequential_9/lstm_1/lstm_cell_1/split_1/ReadVariableOp^sequential_9/lstm_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????????????: : : : : : : : 2\
,sequential_9/dense_18/BiasAdd/ReadVariableOp,sequential_9/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_18/MatMul/ReadVariableOp+sequential_9/dense_18/MatMul/ReadVariableOp2\
,sequential_9/dense_19/BiasAdd/ReadVariableOp,sequential_9/dense_19/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_19/MatMul/ReadVariableOp+sequential_9/dense_19/MatMul/ReadVariableOp2V
)sequential_9/embedding_9/embedding_lookup)sequential_9/embedding_9/embedding_lookup2`
.sequential_9/lstm_1/lstm_cell_1/ReadVariableOp.sequential_9/lstm_1/lstm_cell_1/ReadVariableOp2d
0sequential_9/lstm_1/lstm_cell_1/ReadVariableOp_10sequential_9/lstm_1/lstm_cell_1/ReadVariableOp_12d
0sequential_9/lstm_1/lstm_cell_1/ReadVariableOp_20sequential_9/lstm_1/lstm_cell_1/ReadVariableOp_22d
0sequential_9/lstm_1/lstm_cell_1/ReadVariableOp_30sequential_9/lstm_1/lstm_cell_1/ReadVariableOp_32l
4sequential_9/lstm_1/lstm_cell_1/split/ReadVariableOp4sequential_9/lstm_1/lstm_cell_1/split/ReadVariableOp2p
6sequential_9/lstm_1/lstm_cell_1/split_1/ReadVariableOp6sequential_9/lstm_1/lstm_cell_1/split_1/ReadVariableOp26
sequential_9/lstm_1/whilesequential_9/lstm_1/while:c _
0
_output_shapes
:??????????????????
+
_user_specified_nameembedding_9_input
??
?	
while_body_161817
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
1while_lstm_cell_1_split_readvariableop_resource_0: @A
3while_lstm_cell_1_split_1_readvariableop_resource_0:@=
+while_lstm_cell_1_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
/while_lstm_cell_1_split_readvariableop_resource: @?
1while_lstm_cell_1_split_1_readvariableop_resource:@;
)while_lstm_cell_1_readvariableop_resource:@?? while/lstm_cell_1/ReadVariableOp?"while/lstm_cell_1/ReadVariableOp_1?"while/lstm_cell_1/ReadVariableOp_2?"while/lstm_cell_1/ReadVariableOp_3?&while/lstm_cell_1/split/ReadVariableOp?(while/lstm_cell_1/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype0?
!while/lstm_cell_1/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/ones_likeFill*while/lstm_cell_1/ones_like/Shape:output:0*while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? d
while/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/dropout/MulMul$while/lstm_cell_1/ones_like:output:0(while/lstm_cell_1/dropout/Const:output:0*
T0*'
_output_shapes
:????????? s
while/lstm_cell_1/dropout/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
6while/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_1/dropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2??Nm
(while/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
&while/lstm_cell_1/dropout/GreaterEqualGreaterEqual?while/lstm_cell_1/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_1/dropout/CastCast*while/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
while/lstm_cell_1/dropout/Mul_1Mul!while/lstm_cell_1/dropout/Mul:z:0"while/lstm_cell_1/dropout/Cast:y:0*
T0*'
_output_shapes
:????????? f
!while/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/dropout_1/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_1/Const:output:0*
T0*'
_output_shapes
:????????? u
!while/lstm_cell_1/dropout_1/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_1/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???o
*while/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
 while/lstm_cell_1/dropout_1/CastCast,while/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
!while/lstm_cell_1/dropout_1/Mul_1Mul#while/lstm_cell_1/dropout_1/Mul:z:0$while/lstm_cell_1/dropout_1/Cast:y:0*
T0*'
_output_shapes
:????????? f
!while/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/dropout_2/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_2/Const:output:0*
T0*'
_output_shapes
:????????? u
!while/lstm_cell_1/dropout_2/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_2/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???o
*while/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
 while/lstm_cell_1/dropout_2/CastCast,while/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
!while/lstm_cell_1/dropout_2/Mul_1Mul#while/lstm_cell_1/dropout_2/Mul:z:0$while/lstm_cell_1/dropout_2/Cast:y:0*
T0*'
_output_shapes
:????????? f
!while/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/dropout_3/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_3/Const:output:0*
T0*'
_output_shapes
:????????? u
!while/lstm_cell_1/dropout_3/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_3/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2??'o
*while/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
 while/lstm_cell_1/dropout_3/CastCast,while/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
!while/lstm_cell_1/dropout_3/Mul_1Mul#while/lstm_cell_1/dropout_3/Mul:z:0$while/lstm_cell_1/dropout_3/Cast:y:0*
T0*'
_output_shapes
:????????? f
#while/lstm_cell_1/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:h
#while/lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/ones_like_1Fill,while/lstm_cell_1/ones_like_1/Shape:output:0,while/lstm_cell_1/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????f
!while/lstm_cell_1/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/dropout_4/MulMul&while/lstm_cell_1/ones_like_1:output:0*while/lstm_cell_1/dropout_4/Const:output:0*
T0*'
_output_shapes
:?????????w
!while/lstm_cell_1/dropout_4/ShapeShape&while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_1/dropout_4/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???o
*while/lstm_cell_1/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_1/dropout_4/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_4/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_1/dropout_4/CastCast,while/lstm_cell_1/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
!while/lstm_cell_1/dropout_4/Mul_1Mul#while/lstm_cell_1/dropout_4/Mul:z:0$while/lstm_cell_1/dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????f
!while/lstm_cell_1/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/dropout_5/MulMul&while/lstm_cell_1/ones_like_1:output:0*while/lstm_cell_1/dropout_5/Const:output:0*
T0*'
_output_shapes
:?????????w
!while/lstm_cell_1/dropout_5/ShapeShape&while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_1/dropout_5/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???o
*while/lstm_cell_1/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_1/dropout_5/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_5/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_1/dropout_5/CastCast,while/lstm_cell_1/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
!while/lstm_cell_1/dropout_5/Mul_1Mul#while/lstm_cell_1/dropout_5/Mul:z:0$while/lstm_cell_1/dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????f
!while/lstm_cell_1/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/dropout_6/MulMul&while/lstm_cell_1/ones_like_1:output:0*while/lstm_cell_1/dropout_6/Const:output:0*
T0*'
_output_shapes
:?????????w
!while/lstm_cell_1/dropout_6/ShapeShape&while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_1/dropout_6/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_6/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???o
*while/lstm_cell_1/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_1/dropout_6/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_6/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_1/dropout_6/CastCast,while/lstm_cell_1/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
!while/lstm_cell_1/dropout_6/Mul_1Mul#while/lstm_cell_1/dropout_6/Mul:z:0$while/lstm_cell_1/dropout_6/Cast:y:0*
T0*'
_output_shapes
:?????????f
!while/lstm_cell_1/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/dropout_7/MulMul&while/lstm_cell_1/ones_like_1:output:0*while/lstm_cell_1/dropout_7/Const:output:0*
T0*'
_output_shapes
:?????????w
!while/lstm_cell_1/dropout_7/ShapeShape&while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_1/dropout_7/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_7/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???o
*while/lstm_cell_1/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_1/dropout_7/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_7/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_1/dropout_7/CastCast,while/lstm_cell_1/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
!while/lstm_cell_1/dropout_7/Mul_1Mul#while/lstm_cell_1/dropout_7/Mul:z:0$while/lstm_cell_1/dropout_7/Cast:y:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_1/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_1/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_1/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_1/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_1/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_1/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:????????? c
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_1/split/ReadVariableOpReadVariableOp1while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes

: @*
dtype0?
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0.while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split?
while/lstm_cell_1/MatMulMatMulwhile/lstm_cell_1/mul:z:0 while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/MatMul_1MatMulwhile/lstm_cell_1/mul_1:z:0 while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/MatMul_2MatMulwhile/lstm_cell_1/mul_2:z:0 while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/MatMul_3MatMulwhile/lstm_cell_1/mul_3:z:0 while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????e
#while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
while/lstm_cell_1/split_1Split,while/lstm_cell_1/split_1/split_dim:output:00while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
while/lstm_cell_1/BiasAddBiasAdd"while/lstm_cell_1/MatMul:product:0"while/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/BiasAdd_1BiasAdd$while/lstm_cell_1/MatMul_1:product:0"while/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/BiasAdd_2BiasAdd$while/lstm_cell_1/MatMul_2:product:0"while/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/BiasAdd_3BiasAdd$while/lstm_cell_1/MatMul_3:product:0"while/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_4Mulwhile_placeholder_2%while/lstm_cell_1/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_5Mulwhile_placeholder_2%while/lstm_cell_1/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_6Mulwhile_placeholder_2%while/lstm_cell_1/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_7Mulwhile_placeholder_2%while/lstm_cell_1/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_1/ReadVariableOpReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0v
%while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_1/strided_sliceStridedSlice(while/lstm_cell_1/ReadVariableOp:value:0.while/lstm_cell_1/strided_slice/stack:output:00while/lstm_cell_1/strided_slice/stack_1:output:00while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_1/MatMul_4MatMulwhile/lstm_cell_1/mul_4:z:0(while/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/addAddV2"while/lstm_cell_1/BiasAdd:output:0$while/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????q
while/lstm_cell_1/SigmoidSigmoidwhile/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
"while/lstm_cell_1/ReadVariableOp_1ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0x
'while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_1/strided_slice_1StridedSlice*while/lstm_cell_1/ReadVariableOp_1:value:00while/lstm_cell_1/strided_slice_1/stack:output:02while/lstm_cell_1/strided_slice_1/stack_1:output:02while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_1/MatMul_5MatMulwhile/lstm_cell_1/mul_5:z:0*while/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/add_1AddV2$while/lstm_cell_1/BiasAdd_1:output:0$while/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????u
while/lstm_cell_1/Sigmoid_1Sigmoidwhile/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_8Mulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:??????????
"while/lstm_cell_1/ReadVariableOp_2ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0x
'while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   z
)while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_1/strided_slice_2StridedSlice*while/lstm_cell_1/ReadVariableOp_2:value:00while/lstm_cell_1/strided_slice_2/stack:output:02while/lstm_cell_1/strided_slice_2/stack_1:output:02while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_1/MatMul_6MatMulwhile/lstm_cell_1/mul_6:z:0*while/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/add_2AddV2$while/lstm_cell_1/BiasAdd_2:output:0$while/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????m
while/lstm_cell_1/TanhTanhwhile/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_9Mulwhile/lstm_cell_1/Sigmoid:y:0while/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/add_3AddV2while/lstm_cell_1/mul_8:z:0while/lstm_cell_1/mul_9:z:0*
T0*'
_output_shapes
:??????????
"while/lstm_cell_1/ReadVariableOp_3ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0x
'while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   z
)while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_1/strided_slice_3StridedSlice*while/lstm_cell_1/ReadVariableOp_3:value:00while/lstm_cell_1/strided_slice_3/stack:output:02while/lstm_cell_1/strided_slice_3/stack_1:output:02while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_1/MatMul_7MatMulwhile/lstm_cell_1/mul_7:z:0*while/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/add_4AddV2$while/lstm_cell_1/BiasAdd_3:output:0$while/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????u
while/lstm_cell_1/Sigmoid_2Sigmoidwhile/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????o
while/lstm_cell_1/Tanh_1Tanhwhile/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_10Mulwhile/lstm_cell_1/Sigmoid_2:y:0while/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_10:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_1/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:?????????x
while/Identity_5Identitywhile/lstm_cell_1/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_1_readvariableop_resource+while_lstm_cell_1_readvariableop_resource_0"h
1while_lstm_cell_1_split_1_readvariableop_resource3while_lstm_cell_1_split_1_readvariableop_resource_0"d
/while_lstm_cell_1_split_readvariableop_resource1while_lstm_cell_1_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????:?????????: : : : : 2D
 while/lstm_cell_1/ReadVariableOp while/lstm_cell_1/ReadVariableOp2H
"while/lstm_cell_1/ReadVariableOp_1"while/lstm_cell_1/ReadVariableOp_12H
"while/lstm_cell_1/ReadVariableOp_2"while/lstm_cell_1/ReadVariableOp_22H
"while/lstm_cell_1/ReadVariableOp_3"while/lstm_cell_1/ReadVariableOp_32P
&while/lstm_cell_1/split/ReadVariableOp&while/lstm_cell_1/split/ReadVariableOp2T
(while/lstm_cell_1/split_1/ReadVariableOp(while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
??
?	
while_body_163381
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
1while_lstm_cell_1_split_readvariableop_resource_0: @A
3while_lstm_cell_1_split_1_readvariableop_resource_0:@=
+while_lstm_cell_1_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
/while_lstm_cell_1_split_readvariableop_resource: @?
1while_lstm_cell_1_split_1_readvariableop_resource:@;
)while_lstm_cell_1_readvariableop_resource:@?? while/lstm_cell_1/ReadVariableOp?"while/lstm_cell_1/ReadVariableOp_1?"while/lstm_cell_1/ReadVariableOp_2?"while/lstm_cell_1/ReadVariableOp_3?&while/lstm_cell_1/split/ReadVariableOp?(while/lstm_cell_1/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype0?
!while/lstm_cell_1/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/ones_likeFill*while/lstm_cell_1/ones_like/Shape:output:0*while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? d
while/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/dropout/MulMul$while/lstm_cell_1/ones_like:output:0(while/lstm_cell_1/dropout/Const:output:0*
T0*'
_output_shapes
:????????? s
while/lstm_cell_1/dropout/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
6while/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_1/dropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2ޅ?m
(while/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
&while/lstm_cell_1/dropout/GreaterEqualGreaterEqual?while/lstm_cell_1/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_1/dropout/CastCast*while/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
while/lstm_cell_1/dropout/Mul_1Mul!while/lstm_cell_1/dropout/Mul:z:0"while/lstm_cell_1/dropout/Cast:y:0*
T0*'
_output_shapes
:????????? f
!while/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/dropout_1/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_1/Const:output:0*
T0*'
_output_shapes
:????????? u
!while/lstm_cell_1/dropout_1/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_1/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???o
*while/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
 while/lstm_cell_1/dropout_1/CastCast,while/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
!while/lstm_cell_1/dropout_1/Mul_1Mul#while/lstm_cell_1/dropout_1/Mul:z:0$while/lstm_cell_1/dropout_1/Cast:y:0*
T0*'
_output_shapes
:????????? f
!while/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/dropout_2/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_2/Const:output:0*
T0*'
_output_shapes
:????????? u
!while/lstm_cell_1/dropout_2/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_2/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???o
*while/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
 while/lstm_cell_1/dropout_2/CastCast,while/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
!while/lstm_cell_1/dropout_2/Mul_1Mul#while/lstm_cell_1/dropout_2/Mul:z:0$while/lstm_cell_1/dropout_2/Cast:y:0*
T0*'
_output_shapes
:????????? f
!while/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/dropout_3/MulMul$while/lstm_cell_1/ones_like:output:0*while/lstm_cell_1/dropout_3/Const:output:0*
T0*'
_output_shapes
:????????? u
!while/lstm_cell_1/dropout_3/ShapeShape$while/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_3/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???o
*while/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
 while/lstm_cell_1/dropout_3/CastCast,while/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
!while/lstm_cell_1/dropout_3/Mul_1Mul#while/lstm_cell_1/dropout_3/Mul:z:0$while/lstm_cell_1/dropout_3/Cast:y:0*
T0*'
_output_shapes
:????????? f
#while/lstm_cell_1/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:h
#while/lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/ones_like_1Fill,while/lstm_cell_1/ones_like_1/Shape:output:0,while/lstm_cell_1/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????f
!while/lstm_cell_1/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/dropout_4/MulMul&while/lstm_cell_1/ones_like_1:output:0*while/lstm_cell_1/dropout_4/Const:output:0*
T0*'
_output_shapes
:?????????w
!while/lstm_cell_1/dropout_4/ShapeShape&while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_1/dropout_4/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???o
*while/lstm_cell_1/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_1/dropout_4/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_4/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_1/dropout_4/CastCast,while/lstm_cell_1/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
!while/lstm_cell_1/dropout_4/Mul_1Mul#while/lstm_cell_1/dropout_4/Mul:z:0$while/lstm_cell_1/dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????f
!while/lstm_cell_1/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/dropout_5/MulMul&while/lstm_cell_1/ones_like_1:output:0*while/lstm_cell_1/dropout_5/Const:output:0*
T0*'
_output_shapes
:?????????w
!while/lstm_cell_1/dropout_5/ShapeShape&while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_1/dropout_5/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2ǰyo
*while/lstm_cell_1/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_1/dropout_5/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_5/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_1/dropout_5/CastCast,while/lstm_cell_1/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
!while/lstm_cell_1/dropout_5/Mul_1Mul#while/lstm_cell_1/dropout_5/Mul:z:0$while/lstm_cell_1/dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????f
!while/lstm_cell_1/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/dropout_6/MulMul&while/lstm_cell_1/ones_like_1:output:0*while/lstm_cell_1/dropout_6/Const:output:0*
T0*'
_output_shapes
:?????????w
!while/lstm_cell_1/dropout_6/ShapeShape&while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_1/dropout_6/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_6/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???o
*while/lstm_cell_1/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_1/dropout_6/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_6/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_1/dropout_6/CastCast,while/lstm_cell_1/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
!while/lstm_cell_1/dropout_6/Mul_1Mul#while/lstm_cell_1/dropout_6/Mul:z:0$while/lstm_cell_1/dropout_6/Cast:y:0*
T0*'
_output_shapes
:?????????f
!while/lstm_cell_1/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/dropout_7/MulMul&while/lstm_cell_1/ones_like_1:output:0*while/lstm_cell_1/dropout_7/Const:output:0*
T0*'
_output_shapes
:?????????w
!while/lstm_cell_1/dropout_7/ShapeShape&while/lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_1/dropout_7/random_uniform/RandomUniformRandomUniform*while/lstm_cell_1/dropout_7/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???o
*while/lstm_cell_1/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_1/dropout_7/GreaterEqualGreaterEqualAwhile/lstm_cell_1/dropout_7/random_uniform/RandomUniform:output:03while/lstm_cell_1/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_1/dropout_7/CastCast,while/lstm_cell_1/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
!while/lstm_cell_1/dropout_7/Mul_1Mul#while/lstm_cell_1/dropout_7/Mul:z:0$while/lstm_cell_1/dropout_7/Cast:y:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_1/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_1/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_1/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_1/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_1/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_1/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:????????? c
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_1/split/ReadVariableOpReadVariableOp1while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes

: @*
dtype0?
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0.while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split?
while/lstm_cell_1/MatMulMatMulwhile/lstm_cell_1/mul:z:0 while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/MatMul_1MatMulwhile/lstm_cell_1/mul_1:z:0 while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/MatMul_2MatMulwhile/lstm_cell_1/mul_2:z:0 while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/MatMul_3MatMulwhile/lstm_cell_1/mul_3:z:0 while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????e
#while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
while/lstm_cell_1/split_1Split,while/lstm_cell_1/split_1/split_dim:output:00while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
while/lstm_cell_1/BiasAddBiasAdd"while/lstm_cell_1/MatMul:product:0"while/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/BiasAdd_1BiasAdd$while/lstm_cell_1/MatMul_1:product:0"while/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/BiasAdd_2BiasAdd$while/lstm_cell_1/MatMul_2:product:0"while/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/BiasAdd_3BiasAdd$while/lstm_cell_1/MatMul_3:product:0"while/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_4Mulwhile_placeholder_2%while/lstm_cell_1/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_5Mulwhile_placeholder_2%while/lstm_cell_1/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_6Mulwhile_placeholder_2%while/lstm_cell_1/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_7Mulwhile_placeholder_2%while/lstm_cell_1/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_1/ReadVariableOpReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0v
%while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_1/strided_sliceStridedSlice(while/lstm_cell_1/ReadVariableOp:value:0.while/lstm_cell_1/strided_slice/stack:output:00while/lstm_cell_1/strided_slice/stack_1:output:00while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_1/MatMul_4MatMulwhile/lstm_cell_1/mul_4:z:0(while/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/addAddV2"while/lstm_cell_1/BiasAdd:output:0$while/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????q
while/lstm_cell_1/SigmoidSigmoidwhile/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
"while/lstm_cell_1/ReadVariableOp_1ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0x
'while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_1/strided_slice_1StridedSlice*while/lstm_cell_1/ReadVariableOp_1:value:00while/lstm_cell_1/strided_slice_1/stack:output:02while/lstm_cell_1/strided_slice_1/stack_1:output:02while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_1/MatMul_5MatMulwhile/lstm_cell_1/mul_5:z:0*while/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/add_1AddV2$while/lstm_cell_1/BiasAdd_1:output:0$while/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????u
while/lstm_cell_1/Sigmoid_1Sigmoidwhile/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_8Mulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:??????????
"while/lstm_cell_1/ReadVariableOp_2ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0x
'while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   z
)while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_1/strided_slice_2StridedSlice*while/lstm_cell_1/ReadVariableOp_2:value:00while/lstm_cell_1/strided_slice_2/stack:output:02while/lstm_cell_1/strided_slice_2/stack_1:output:02while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_1/MatMul_6MatMulwhile/lstm_cell_1/mul_6:z:0*while/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/add_2AddV2$while/lstm_cell_1/BiasAdd_2:output:0$while/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????m
while/lstm_cell_1/TanhTanhwhile/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_9Mulwhile/lstm_cell_1/Sigmoid:y:0while/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/add_3AddV2while/lstm_cell_1/mul_8:z:0while/lstm_cell_1/mul_9:z:0*
T0*'
_output_shapes
:??????????
"while/lstm_cell_1/ReadVariableOp_3ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0x
'while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   z
)while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_1/strided_slice_3StridedSlice*while/lstm_cell_1/ReadVariableOp_3:value:00while/lstm_cell_1/strided_slice_3/stack:output:02while/lstm_cell_1/strided_slice_3/stack_1:output:02while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_1/MatMul_7MatMulwhile/lstm_cell_1/mul_7:z:0*while/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/add_4AddV2$while/lstm_cell_1/BiasAdd_3:output:0$while/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????u
while/lstm_cell_1/Sigmoid_2Sigmoidwhile/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????o
while/lstm_cell_1/Tanh_1Tanhwhile/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_10Mulwhile/lstm_cell_1/Sigmoid_2:y:0while/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_10:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_1/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:?????????x
while/Identity_5Identitywhile/lstm_cell_1/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_1_readvariableop_resource+while_lstm_cell_1_readvariableop_resource_0"h
1while_lstm_cell_1_split_1_readvariableop_resource3while_lstm_cell_1_split_1_readvariableop_resource_0"d
/while_lstm_cell_1_split_readvariableop_resource1while_lstm_cell_1_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????:?????????: : : : : 2D
 while/lstm_cell_1/ReadVariableOp while/lstm_cell_1/ReadVariableOp2H
"while/lstm_cell_1/ReadVariableOp_1"while/lstm_cell_1/ReadVariableOp_12H
"while/lstm_cell_1/ReadVariableOp_2"while/lstm_cell_1/ReadVariableOp_22H
"while/lstm_cell_1/ReadVariableOp_3"while/lstm_cell_1/ReadVariableOp_32P
&while/lstm_cell_1/split/ReadVariableOp&while/lstm_cell_1/split/ReadVariableOp2T
(while/lstm_cell_1/split_1/ReadVariableOp(while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_161816
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_161816___redundant_placeholder04
0while_while_cond_161816___redundant_placeholder14
0while_while_cond_161816___redundant_placeholder24
0while_while_cond_161816___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_162080

inputs%
embedding_9_162058:	?' 
lstm_1_162061: @
lstm_1_162063:@
lstm_1_162065:@!
dense_18_162068:

dense_18_162070:
!
dense_19_162074:

dense_19_162076:
identity?? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall?!dropout_9/StatefulPartitionedCall?#embedding_9/StatefulPartitionedCall?lstm_1/StatefulPartitionedCall?
#embedding_9/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_9_162058*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_9_layer_call_and_return_conditional_losses_161273?
lstm_1/StatefulPartitionedCallStatefulPartitionedCall,embedding_9/StatefulPartitionedCall:output:0lstm_1_162061lstm_1_162063lstm_1_162065*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_162015?
 dense_18/StatefulPartitionedCallStatefulPartitionedCall'lstm_1/StatefulPartitionedCall:output:0dense_18_162068dense_18_162070*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_161538?
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_161618?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0dense_19_162074dense_19_162076*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_161562x
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall$^embedding_9/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????????????: : : : : : : : 2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2J
#embedding_9/StatefulPartitionedCall#embedding_9/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
while_cond_163380
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_163380___redundant_placeholder04
0while_while_cond_163380___redundant_placeholder14
0while_while_cond_163380___redundant_placeholder24
0while_while_cond_163380___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
??
?
B__inference_lstm_1_layer_call_and_return_conditional_losses_161519

inputs;
)lstm_cell_1_split_readvariableop_resource: @9
+lstm_cell_1_split_1_readvariableop_resource:@5
#lstm_cell_1_readvariableop_resource:@
identity??lstm_cell_1/ReadVariableOp?lstm_cell_1/ReadVariableOp_1?lstm_cell_1/ReadVariableOp_2?lstm_cell_1/ReadVariableOp_3? lstm_cell_1/split/ReadVariableOp?"lstm_cell_1/split_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_maskc
lstm_cell_1/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:`
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? [
lstm_cell_1/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:b
lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/ones_like_1Fill&lstm_cell_1/ones_like_1/Shape:output:0&lstm_cell_1/ones_like_1/Const:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/mulMulstrided_slice_2:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/mul_1Mulstrided_slice_2:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/mul_2Mulstrided_slice_2:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/mul_3Mulstrided_slice_2:output:0lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_1/split/ReadVariableOpReadVariableOp)lstm_cell_1_split_readvariableop_resource*
_output_shapes

: @*
dtype0?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0(lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split
lstm_cell_1/MatMulMatMullstm_cell_1/mul:z:0lstm_cell_1/split:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/MatMul_1MatMullstm_cell_1/mul_1:z:0lstm_cell_1/split:output:1*
T0*'
_output_shapes
:??????????
lstm_cell_1/MatMul_2MatMullstm_cell_1/mul_2:z:0lstm_cell_1/split:output:2*
T0*'
_output_shapes
:??????????
lstm_cell_1/MatMul_3MatMullstm_cell_1/mul_3:z:0lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????_
lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_1/split_1/ReadVariableOpReadVariableOp+lstm_cell_1_split_1_readvariableop_resource*
_output_shapes
:@*
dtype0?
lstm_cell_1/split_1Split&lstm_cell_1/split_1/split_dim:output:0*lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/MatMul:product:0lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/BiasAdd_1BiasAddlstm_cell_1/MatMul_1:product:0lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:??????????
lstm_cell_1/BiasAdd_2BiasAddlstm_cell_1/MatMul_2:product:0lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
lstm_cell_1/BiasAdd_3BiasAddlstm_cell_1/MatMul_3:product:0lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????|
lstm_cell_1/mul_4Mulzeros:output:0 lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:?????????|
lstm_cell_1/mul_5Mulzeros:output:0 lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:?????????|
lstm_cell_1/mul_6Mulzeros:output:0 lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:?????????|
lstm_cell_1/mul_7Mulzeros:output:0 lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:?????????~
lstm_cell_1/ReadVariableOpReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0p
lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_1/strided_sliceStridedSlice"lstm_cell_1/ReadVariableOp:value:0(lstm_cell_1/strided_slice/stack:output:0*lstm_cell_1/strided_slice/stack_1:output:0*lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_1/MatMul_4MatMullstm_cell_1/mul_4:z:0"lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/addAddV2lstm_cell_1/BiasAdd:output:0lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????e
lstm_cell_1/SigmoidSigmoidlstm_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/ReadVariableOp_1ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0r
!lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_1/strided_slice_1StridedSlice$lstm_cell_1/ReadVariableOp_1:value:0*lstm_cell_1/strided_slice_1/stack:output:0,lstm_cell_1/strided_slice_1/stack_1:output:0,lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_1/MatMul_5MatMullstm_cell_1/mul_5:z:0$lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/add_1AddV2lstm_cell_1/BiasAdd_1:output:0lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????i
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????w
lstm_cell_1/mul_8Mullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/ReadVariableOp_2ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0r
!lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   t
#lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_1/strided_slice_2StridedSlice$lstm_cell_1/ReadVariableOp_2:value:0*lstm_cell_1/strided_slice_2/stack:output:0,lstm_cell_1/strided_slice_2/stack_1:output:0,lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_1/MatMul_6MatMullstm_cell_1/mul_6:z:0$lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/add_2AddV2lstm_cell_1/BiasAdd_2:output:0lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????a
lstm_cell_1/TanhTanhlstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????y
lstm_cell_1/mul_9Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????z
lstm_cell_1/add_3AddV2lstm_cell_1/mul_8:z:0lstm_cell_1/mul_9:z:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/ReadVariableOp_3ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0r
!lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   t
#lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_1/strided_slice_3StridedSlice$lstm_cell_1/ReadVariableOp_3:value:0*lstm_cell_1/strided_slice_3/stack:output:0,lstm_cell_1/strided_slice_3/stack_1:output:0,lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_1/MatMul_7MatMullstm_cell_1/mul_7:z:0$lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/add_4AddV2lstm_cell_1/BiasAdd_3:output:0lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????i
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????c
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????~
lstm_cell_1/mul_10Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_1_split_readvariableop_resource+lstm_cell_1_split_1_readvariableop_resource#lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_161385*
condR
while_cond_161384*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????????????? : : : 28
lstm_cell_1/ReadVariableOplstm_cell_1/ReadVariableOp2<
lstm_cell_1/ReadVariableOp_1lstm_cell_1/ReadVariableOp_12<
lstm_cell_1/ReadVariableOp_2lstm_cell_1/ReadVariableOp_22<
lstm_cell_1/ReadVariableOp_3lstm_cell_1/ReadVariableOp_32D
 lstm_cell_1/split/ReadVariableOp lstm_cell_1/split/ReadVariableOp2H
"lstm_cell_1/split_1/ReadVariableOp"lstm_cell_1/split_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?
?
while_cond_163073
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_163073___redundant_placeholder04
0while_while_cond_163073___redundant_placeholder14
0while_while_cond_163073___redundant_placeholder24
0while_while_cond_163073___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
c
E__inference_dropout_9_layer_call_and_return_conditional_losses_161549

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????
[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?	
d
E__inference_dropout_9_layer_call_and_return_conditional_losses_161618

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????
o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????
Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
while_cond_163994
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_163994___redundant_placeholder04
0while_while_cond_163994___redundant_placeholder14
0while_while_cond_163994___redundant_placeholder24
0while_while_cond_163994___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?D
?
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_160859

inputs

states
states_1/
split_readvariableop_resource: @-
split_1_readvariableop_resource:@)
readvariableop_resource:@
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:????????? G
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??}
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????X
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:????????? Z
mul_1Mulinputsones_like:output:0*
T0*'
_output_shapes
:????????? Z
mul_2Mulinputsones_like:output:0*
T0*'
_output_shapes
:????????? Z
mul_3Mulinputsones_like:output:0*
T0*'
_output_shapes
:????????? Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :r
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

: @*
dtype0?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split[
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:?????????_
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:?????????_
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:?????????_
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:?????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:@*
dtype0?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????\
mul_4Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:?????????\
mul_5Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:?????????\
mul_6Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:?????????\
mul_7Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:?????????f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskg
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:?????????d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:?????????h
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????W
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????h
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????U
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????V
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:?????????h
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????h
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????Q
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:?????????K
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:?????????Z
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????Y
IdentityIdentity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:?????????[

Identity_1Identity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:?????????Z

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:????????? :?????????:?????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_namestates:OK
'
_output_shapes
:?????????
 
_user_specified_namestates
?
?
while_cond_160872
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_160872___redundant_placeholder04
0while_while_cond_160872___redundant_placeholder14
0while_while_cond_160872___redundant_placeholder24
0while_while_cond_160872___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_161177
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_161177___redundant_placeholder04
0while_while_cond_161177___redundant_placeholder14
0while_while_cond_161177___redundant_placeholder24
0while_while_cond_161177___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
c
*__inference_dropout_9_layer_call_fn_164223

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_161618o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????
22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?"
?
while_body_160873
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_1_160897_0: @(
while_lstm_cell_1_160899_0:@,
while_lstm_cell_1_160901_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_1_160897: @&
while_lstm_cell_1_160899:@*
while_lstm_cell_1_160901:@??)while/lstm_cell_1/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype0?
)while/lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_1_160897_0while_lstm_cell_1_160899_0while_lstm_cell_1_160901_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_160859?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity2while/lstm_cell_1/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:??????????
while/Identity_5Identity2while/lstm_cell_1/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:?????????x

while/NoOpNoOp*^while/lstm_cell_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_1_160897while_lstm_cell_1_160897_0"6
while_lstm_cell_1_160899while_lstm_cell_1_160899_0"6
while_lstm_cell_1_160901while_lstm_cell_1_160901_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????:?????????: : : : : 2V
)while/lstm_cell_1/StatefulPartitionedCall)while/lstm_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?t
?	
while_body_163074
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
1while_lstm_cell_1_split_readvariableop_resource_0: @A
3while_lstm_cell_1_split_1_readvariableop_resource_0:@=
+while_lstm_cell_1_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
/while_lstm_cell_1_split_readvariableop_resource: @?
1while_lstm_cell_1_split_1_readvariableop_resource:@;
)while_lstm_cell_1_readvariableop_resource:@?? while/lstm_cell_1/ReadVariableOp?"while/lstm_cell_1/ReadVariableOp_1?"while/lstm_cell_1/ReadVariableOp_2?"while/lstm_cell_1/ReadVariableOp_3?&while/lstm_cell_1/split/ReadVariableOp?(while/lstm_cell_1/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype0?
!while/lstm_cell_1/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/ones_likeFill*while/lstm_cell_1/ones_like/Shape:output:0*while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? f
#while/lstm_cell_1/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:h
#while/lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/ones_like_1Fill,while/lstm_cell_1/ones_like_1/Shape:output:0,while/lstm_cell_1/ones_like_1/Const:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_1/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_1/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_1/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? c
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_1/split/ReadVariableOpReadVariableOp1while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes

: @*
dtype0?
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0.while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split?
while/lstm_cell_1/MatMulMatMulwhile/lstm_cell_1/mul:z:0 while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/MatMul_1MatMulwhile/lstm_cell_1/mul_1:z:0 while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/MatMul_2MatMulwhile/lstm_cell_1/mul_2:z:0 while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/MatMul_3MatMulwhile/lstm_cell_1/mul_3:z:0 while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????e
#while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
while/lstm_cell_1/split_1Split,while/lstm_cell_1/split_1/split_dim:output:00while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
while/lstm_cell_1/BiasAddBiasAdd"while/lstm_cell_1/MatMul:product:0"while/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/BiasAdd_1BiasAdd$while/lstm_cell_1/MatMul_1:product:0"while/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/BiasAdd_2BiasAdd$while/lstm_cell_1/MatMul_2:product:0"while/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/BiasAdd_3BiasAdd$while/lstm_cell_1/MatMul_3:product:0"while/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_4Mulwhile_placeholder_2&while/lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_5Mulwhile_placeholder_2&while/lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_6Mulwhile_placeholder_2&while/lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_7Mulwhile_placeholder_2&while/lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_1/ReadVariableOpReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0v
%while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_1/strided_sliceStridedSlice(while/lstm_cell_1/ReadVariableOp:value:0.while/lstm_cell_1/strided_slice/stack:output:00while/lstm_cell_1/strided_slice/stack_1:output:00while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_1/MatMul_4MatMulwhile/lstm_cell_1/mul_4:z:0(while/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/addAddV2"while/lstm_cell_1/BiasAdd:output:0$while/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????q
while/lstm_cell_1/SigmoidSigmoidwhile/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
"while/lstm_cell_1/ReadVariableOp_1ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0x
'while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_1/strided_slice_1StridedSlice*while/lstm_cell_1/ReadVariableOp_1:value:00while/lstm_cell_1/strided_slice_1/stack:output:02while/lstm_cell_1/strided_slice_1/stack_1:output:02while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_1/MatMul_5MatMulwhile/lstm_cell_1/mul_5:z:0*while/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/add_1AddV2$while/lstm_cell_1/BiasAdd_1:output:0$while/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????u
while/lstm_cell_1/Sigmoid_1Sigmoidwhile/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_8Mulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:??????????
"while/lstm_cell_1/ReadVariableOp_2ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0x
'while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   z
)while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_1/strided_slice_2StridedSlice*while/lstm_cell_1/ReadVariableOp_2:value:00while/lstm_cell_1/strided_slice_2/stack:output:02while/lstm_cell_1/strided_slice_2/stack_1:output:02while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_1/MatMul_6MatMulwhile/lstm_cell_1/mul_6:z:0*while/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/add_2AddV2$while/lstm_cell_1/BiasAdd_2:output:0$while/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????m
while/lstm_cell_1/TanhTanhwhile/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_9Mulwhile/lstm_cell_1/Sigmoid:y:0while/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/add_3AddV2while/lstm_cell_1/mul_8:z:0while/lstm_cell_1/mul_9:z:0*
T0*'
_output_shapes
:??????????
"while/lstm_cell_1/ReadVariableOp_3ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0x
'while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   z
)while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_1/strided_slice_3StridedSlice*while/lstm_cell_1/ReadVariableOp_3:value:00while/lstm_cell_1/strided_slice_3/stack:output:02while/lstm_cell_1/strided_slice_3/stack_1:output:02while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_1/MatMul_7MatMulwhile/lstm_cell_1/mul_7:z:0*while/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/add_4AddV2$while/lstm_cell_1/BiasAdd_3:output:0$while/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????u
while/lstm_cell_1/Sigmoid_2Sigmoidwhile/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????o
while/lstm_cell_1/Tanh_1Tanhwhile/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_10Mulwhile/lstm_cell_1/Sigmoid_2:y:0while/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_10:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_1/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:?????????x
while/Identity_5Identitywhile/lstm_cell_1/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_1_readvariableop_resource+while_lstm_cell_1_readvariableop_resource_0"h
1while_lstm_cell_1_split_1_readvariableop_resource3while_lstm_cell_1_split_1_readvariableop_resource_0"d
/while_lstm_cell_1_split_readvariableop_resource1while_lstm_cell_1_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????:?????????: : : : : 2D
 while/lstm_cell_1/ReadVariableOp while/lstm_cell_1/ReadVariableOp2H
"while/lstm_cell_1/ReadVariableOp_1"while/lstm_cell_1/ReadVariableOp_12H
"while/lstm_cell_1/ReadVariableOp_2"while/lstm_cell_1/ReadVariableOp_22H
"while/lstm_cell_1/ReadVariableOp_3"while/lstm_cell_1/ReadVariableOp_32P
&while/lstm_cell_1/split/ReadVariableOp&while/lstm_cell_1/split/ReadVariableOp2T
(while/lstm_cell_1/split_1/ReadVariableOp(while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?t
?	
while_body_163688
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
1while_lstm_cell_1_split_readvariableop_resource_0: @A
3while_lstm_cell_1_split_1_readvariableop_resource_0:@=
+while_lstm_cell_1_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
/while_lstm_cell_1_split_readvariableop_resource: @?
1while_lstm_cell_1_split_1_readvariableop_resource:@;
)while_lstm_cell_1_readvariableop_resource:@?? while/lstm_cell_1/ReadVariableOp?"while/lstm_cell_1/ReadVariableOp_1?"while/lstm_cell_1/ReadVariableOp_2?"while/lstm_cell_1/ReadVariableOp_3?&while/lstm_cell_1/split/ReadVariableOp?(while/lstm_cell_1/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype0?
!while/lstm_cell_1/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/ones_likeFill*while/lstm_cell_1/ones_like/Shape:output:0*while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? f
#while/lstm_cell_1/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:h
#while/lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_1/ones_like_1Fill,while/lstm_cell_1/ones_like_1/Shape:output:0,while/lstm_cell_1/ones_like_1/Const:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_1/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_1/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
while/lstm_cell_1/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? c
!while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_1/split/ReadVariableOpReadVariableOp1while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes

: @*
dtype0?
while/lstm_cell_1/splitSplit*while/lstm_cell_1/split/split_dim:output:0.while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split?
while/lstm_cell_1/MatMulMatMulwhile/lstm_cell_1/mul:z:0 while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/MatMul_1MatMulwhile/lstm_cell_1/mul_1:z:0 while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/MatMul_2MatMulwhile/lstm_cell_1/mul_2:z:0 while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/MatMul_3MatMulwhile/lstm_cell_1/mul_3:z:0 while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????e
#while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
while/lstm_cell_1/split_1Split,while/lstm_cell_1/split_1/split_dim:output:00while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
while/lstm_cell_1/BiasAddBiasAdd"while/lstm_cell_1/MatMul:product:0"while/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/BiasAdd_1BiasAdd$while/lstm_cell_1/MatMul_1:product:0"while/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/BiasAdd_2BiasAdd$while/lstm_cell_1/MatMul_2:product:0"while/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/BiasAdd_3BiasAdd$while/lstm_cell_1/MatMul_3:product:0"while/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_4Mulwhile_placeholder_2&while/lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_5Mulwhile_placeholder_2&while/lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_6Mulwhile_placeholder_2&while/lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_7Mulwhile_placeholder_2&while/lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
 while/lstm_cell_1/ReadVariableOpReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0v
%while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_1/strided_sliceStridedSlice(while/lstm_cell_1/ReadVariableOp:value:0.while/lstm_cell_1/strided_slice/stack:output:00while/lstm_cell_1/strided_slice/stack_1:output:00while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_1/MatMul_4MatMulwhile/lstm_cell_1/mul_4:z:0(while/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/addAddV2"while/lstm_cell_1/BiasAdd:output:0$while/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????q
while/lstm_cell_1/SigmoidSigmoidwhile/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
"while/lstm_cell_1/ReadVariableOp_1ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0x
'while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_1/strided_slice_1StridedSlice*while/lstm_cell_1/ReadVariableOp_1:value:00while/lstm_cell_1/strided_slice_1/stack:output:02while/lstm_cell_1/strided_slice_1/stack_1:output:02while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_1/MatMul_5MatMulwhile/lstm_cell_1/mul_5:z:0*while/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/add_1AddV2$while/lstm_cell_1/BiasAdd_1:output:0$while/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????u
while/lstm_cell_1/Sigmoid_1Sigmoidwhile/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_8Mulwhile/lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:??????????
"while/lstm_cell_1/ReadVariableOp_2ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0x
'while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   z
)while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_1/strided_slice_2StridedSlice*while/lstm_cell_1/ReadVariableOp_2:value:00while/lstm_cell_1/strided_slice_2/stack:output:02while/lstm_cell_1/strided_slice_2/stack_1:output:02while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_1/MatMul_6MatMulwhile/lstm_cell_1/mul_6:z:0*while/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/add_2AddV2$while/lstm_cell_1/BiasAdd_2:output:0$while/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????m
while/lstm_cell_1/TanhTanhwhile/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_9Mulwhile/lstm_cell_1/Sigmoid:y:0while/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/add_3AddV2while/lstm_cell_1/mul_8:z:0while/lstm_cell_1/mul_9:z:0*
T0*'
_output_shapes
:??????????
"while/lstm_cell_1/ReadVariableOp_3ReadVariableOp+while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0x
'while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   z
)while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_1/strided_slice_3StridedSlice*while/lstm_cell_1/ReadVariableOp_3:value:00while/lstm_cell_1/strided_slice_3/stack:output:02while/lstm_cell_1/strided_slice_3/stack_1:output:02while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
while/lstm_cell_1/MatMul_7MatMulwhile/lstm_cell_1/mul_7:z:0*while/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/add_4AddV2$while/lstm_cell_1/BiasAdd_3:output:0$while/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????u
while/lstm_cell_1/Sigmoid_2Sigmoidwhile/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????o
while/lstm_cell_1/Tanh_1Tanhwhile/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:??????????
while/lstm_cell_1/mul_10Mulwhile/lstm_cell_1/Sigmoid_2:y:0while/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_1/mul_10:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_1/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:?????????x
while/Identity_5Identitywhile/lstm_cell_1/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp!^while/lstm_cell_1/ReadVariableOp#^while/lstm_cell_1/ReadVariableOp_1#^while/lstm_cell_1/ReadVariableOp_2#^while/lstm_cell_1/ReadVariableOp_3'^while/lstm_cell_1/split/ReadVariableOp)^while/lstm_cell_1/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_1_readvariableop_resource+while_lstm_cell_1_readvariableop_resource_0"h
1while_lstm_cell_1_split_1_readvariableop_resource3while_lstm_cell_1_split_1_readvariableop_resource_0"d
/while_lstm_cell_1_split_readvariableop_resource1while_lstm_cell_1_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????:?????????: : : : : 2D
 while/lstm_cell_1/ReadVariableOp while/lstm_cell_1/ReadVariableOp2H
"while/lstm_cell_1/ReadVariableOp_1"while/lstm_cell_1/ReadVariableOp_12H
"while/lstm_cell_1/ReadVariableOp_2"while/lstm_cell_1/ReadVariableOp_22H
"while/lstm_cell_1/ReadVariableOp_3"while/lstm_cell_1/ReadVariableOp_32P
&while/lstm_cell_1/split/ReadVariableOp&while/lstm_cell_1/split/ReadVariableOp2T
(while/lstm_cell_1/split_1/ReadVariableOp(while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_161384
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_161384___redundant_placeholder04
0while_while_cond_161384___redundant_placeholder14
0while_while_cond_161384___redundant_placeholder24
0while_while_cond_161384___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
Ο
?
%sequential_9_lstm_1_while_body_160593D
@sequential_9_lstm_1_while_sequential_9_lstm_1_while_loop_counterJ
Fsequential_9_lstm_1_while_sequential_9_lstm_1_while_maximum_iterations)
%sequential_9_lstm_1_while_placeholder+
'sequential_9_lstm_1_while_placeholder_1+
'sequential_9_lstm_1_while_placeholder_2+
'sequential_9_lstm_1_while_placeholder_3C
?sequential_9_lstm_1_while_sequential_9_lstm_1_strided_slice_1_0
{sequential_9_lstm_1_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_1_tensorarrayunstack_tensorlistfromtensor_0W
Esequential_9_lstm_1_while_lstm_cell_1_split_readvariableop_resource_0: @U
Gsequential_9_lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_0:@Q
?sequential_9_lstm_1_while_lstm_cell_1_readvariableop_resource_0:@&
"sequential_9_lstm_1_while_identity(
$sequential_9_lstm_1_while_identity_1(
$sequential_9_lstm_1_while_identity_2(
$sequential_9_lstm_1_while_identity_3(
$sequential_9_lstm_1_while_identity_4(
$sequential_9_lstm_1_while_identity_5A
=sequential_9_lstm_1_while_sequential_9_lstm_1_strided_slice_1}
ysequential_9_lstm_1_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_1_tensorarrayunstack_tensorlistfromtensorU
Csequential_9_lstm_1_while_lstm_cell_1_split_readvariableop_resource: @S
Esequential_9_lstm_1_while_lstm_cell_1_split_1_readvariableop_resource:@O
=sequential_9_lstm_1_while_lstm_cell_1_readvariableop_resource:@??4sequential_9/lstm_1/while/lstm_cell_1/ReadVariableOp?6sequential_9/lstm_1/while/lstm_cell_1/ReadVariableOp_1?6sequential_9/lstm_1/while/lstm_cell_1/ReadVariableOp_2?6sequential_9/lstm_1/while/lstm_cell_1/ReadVariableOp_3?:sequential_9/lstm_1/while/lstm_cell_1/split/ReadVariableOp?<sequential_9/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp?
Ksequential_9/lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
=sequential_9/lstm_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_9_lstm_1_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_1_tensorarrayunstack_tensorlistfromtensor_0%sequential_9_lstm_1_while_placeholderTsequential_9/lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype0?
5sequential_9/lstm_1/while/lstm_cell_1/ones_like/ShapeShapeDsequential_9/lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:z
5sequential_9/lstm_1/while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
/sequential_9/lstm_1/while/lstm_cell_1/ones_likeFill>sequential_9/lstm_1/while/lstm_cell_1/ones_like/Shape:output:0>sequential_9/lstm_1/while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? ?
7sequential_9/lstm_1/while/lstm_cell_1/ones_like_1/ShapeShape'sequential_9_lstm_1_while_placeholder_2*
T0*
_output_shapes
:|
7sequential_9/lstm_1/while/lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
1sequential_9/lstm_1/while/lstm_cell_1/ones_like_1Fill@sequential_9/lstm_1/while/lstm_cell_1/ones_like_1/Shape:output:0@sequential_9/lstm_1/while/lstm_cell_1/ones_like_1/Const:output:0*
T0*'
_output_shapes
:??????????
)sequential_9/lstm_1/while/lstm_cell_1/mulMulDsequential_9/lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:08sequential_9/lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
+sequential_9/lstm_1/while/lstm_cell_1/mul_1MulDsequential_9/lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:08sequential_9/lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
+sequential_9/lstm_1/while/lstm_cell_1/mul_2MulDsequential_9/lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:08sequential_9/lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
+sequential_9/lstm_1/while/lstm_cell_1/mul_3MulDsequential_9/lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:08sequential_9/lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? w
5sequential_9/lstm_1/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
:sequential_9/lstm_1/while/lstm_cell_1/split/ReadVariableOpReadVariableOpEsequential_9_lstm_1_while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes

: @*
dtype0?
+sequential_9/lstm_1/while/lstm_cell_1/splitSplit>sequential_9/lstm_1/while/lstm_cell_1/split/split_dim:output:0Bsequential_9/lstm_1/while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split?
,sequential_9/lstm_1/while/lstm_cell_1/MatMulMatMul-sequential_9/lstm_1/while/lstm_cell_1/mul:z:04sequential_9/lstm_1/while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:??????????
.sequential_9/lstm_1/while/lstm_cell_1/MatMul_1MatMul/sequential_9/lstm_1/while/lstm_cell_1/mul_1:z:04sequential_9/lstm_1/while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:??????????
.sequential_9/lstm_1/while/lstm_cell_1/MatMul_2MatMul/sequential_9/lstm_1/while/lstm_cell_1/mul_2:z:04sequential_9/lstm_1/while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:??????????
.sequential_9/lstm_1/while/lstm_cell_1/MatMul_3MatMul/sequential_9/lstm_1/while/lstm_cell_1/mul_3:z:04sequential_9/lstm_1/while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????y
7sequential_9/lstm_1/while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
<sequential_9/lstm_1/while/lstm_cell_1/split_1/ReadVariableOpReadVariableOpGsequential_9_lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
-sequential_9/lstm_1/while/lstm_cell_1/split_1Split@sequential_9/lstm_1/while/lstm_cell_1/split_1/split_dim:output:0Dsequential_9/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
-sequential_9/lstm_1/while/lstm_cell_1/BiasAddBiasAdd6sequential_9/lstm_1/while/lstm_cell_1/MatMul:product:06sequential_9/lstm_1/while/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:??????????
/sequential_9/lstm_1/while/lstm_cell_1/BiasAdd_1BiasAdd8sequential_9/lstm_1/while/lstm_cell_1/MatMul_1:product:06sequential_9/lstm_1/while/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:??????????
/sequential_9/lstm_1/while/lstm_cell_1/BiasAdd_2BiasAdd8sequential_9/lstm_1/while/lstm_cell_1/MatMul_2:product:06sequential_9/lstm_1/while/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
/sequential_9/lstm_1/while/lstm_cell_1/BiasAdd_3BiasAdd8sequential_9/lstm_1/while/lstm_cell_1/MatMul_3:product:06sequential_9/lstm_1/while/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:??????????
+sequential_9/lstm_1/while/lstm_cell_1/mul_4Mul'sequential_9_lstm_1_while_placeholder_2:sequential_9/lstm_1/while/lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
+sequential_9/lstm_1/while/lstm_cell_1/mul_5Mul'sequential_9_lstm_1_while_placeholder_2:sequential_9/lstm_1/while/lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
+sequential_9/lstm_1/while/lstm_cell_1/mul_6Mul'sequential_9_lstm_1_while_placeholder_2:sequential_9/lstm_1/while/lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
+sequential_9/lstm_1/while/lstm_cell_1/mul_7Mul'sequential_9_lstm_1_while_placeholder_2:sequential_9/lstm_1/while/lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
4sequential_9/lstm_1/while/lstm_cell_1/ReadVariableOpReadVariableOp?sequential_9_lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0?
9sequential_9/lstm_1/while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
;sequential_9/lstm_1/while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
;sequential_9/lstm_1/while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
3sequential_9/lstm_1/while/lstm_cell_1/strided_sliceStridedSlice<sequential_9/lstm_1/while/lstm_cell_1/ReadVariableOp:value:0Bsequential_9/lstm_1/while/lstm_cell_1/strided_slice/stack:output:0Dsequential_9/lstm_1/while/lstm_cell_1/strided_slice/stack_1:output:0Dsequential_9/lstm_1/while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
.sequential_9/lstm_1/while/lstm_cell_1/MatMul_4MatMul/sequential_9/lstm_1/while/lstm_cell_1/mul_4:z:0<sequential_9/lstm_1/while/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:??????????
)sequential_9/lstm_1/while/lstm_cell_1/addAddV26sequential_9/lstm_1/while/lstm_cell_1/BiasAdd:output:08sequential_9/lstm_1/while/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:??????????
-sequential_9/lstm_1/while/lstm_cell_1/SigmoidSigmoid-sequential_9/lstm_1/while/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
6sequential_9/lstm_1/while/lstm_cell_1/ReadVariableOp_1ReadVariableOp?sequential_9_lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0?
;sequential_9/lstm_1/while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       ?
=sequential_9/lstm_1/while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
=sequential_9/lstm_1/while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
5sequential_9/lstm_1/while/lstm_cell_1/strided_slice_1StridedSlice>sequential_9/lstm_1/while/lstm_cell_1/ReadVariableOp_1:value:0Dsequential_9/lstm_1/while/lstm_cell_1/strided_slice_1/stack:output:0Fsequential_9/lstm_1/while/lstm_cell_1/strided_slice_1/stack_1:output:0Fsequential_9/lstm_1/while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
.sequential_9/lstm_1/while/lstm_cell_1/MatMul_5MatMul/sequential_9/lstm_1/while/lstm_cell_1/mul_5:z:0>sequential_9/lstm_1/while/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
+sequential_9/lstm_1/while/lstm_cell_1/add_1AddV28sequential_9/lstm_1/while/lstm_cell_1/BiasAdd_1:output:08sequential_9/lstm_1/while/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:??????????
/sequential_9/lstm_1/while/lstm_cell_1/Sigmoid_1Sigmoid/sequential_9/lstm_1/while/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:??????????
+sequential_9/lstm_1/while/lstm_cell_1/mul_8Mul3sequential_9/lstm_1/while/lstm_cell_1/Sigmoid_1:y:0'sequential_9_lstm_1_while_placeholder_3*
T0*'
_output_shapes
:??????????
6sequential_9/lstm_1/while/lstm_cell_1/ReadVariableOp_2ReadVariableOp?sequential_9_lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0?
;sequential_9/lstm_1/while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
=sequential_9/lstm_1/while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   ?
=sequential_9/lstm_1/while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
5sequential_9/lstm_1/while/lstm_cell_1/strided_slice_2StridedSlice>sequential_9/lstm_1/while/lstm_cell_1/ReadVariableOp_2:value:0Dsequential_9/lstm_1/while/lstm_cell_1/strided_slice_2/stack:output:0Fsequential_9/lstm_1/while/lstm_cell_1/strided_slice_2/stack_1:output:0Fsequential_9/lstm_1/while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
.sequential_9/lstm_1/while/lstm_cell_1/MatMul_6MatMul/sequential_9/lstm_1/while/lstm_cell_1/mul_6:z:0>sequential_9/lstm_1/while/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
+sequential_9/lstm_1/while/lstm_cell_1/add_2AddV28sequential_9/lstm_1/while/lstm_cell_1/BiasAdd_2:output:08sequential_9/lstm_1/while/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:??????????
*sequential_9/lstm_1/while/lstm_cell_1/TanhTanh/sequential_9/lstm_1/while/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:??????????
+sequential_9/lstm_1/while/lstm_cell_1/mul_9Mul1sequential_9/lstm_1/while/lstm_cell_1/Sigmoid:y:0.sequential_9/lstm_1/while/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:??????????
+sequential_9/lstm_1/while/lstm_cell_1/add_3AddV2/sequential_9/lstm_1/while/lstm_cell_1/mul_8:z:0/sequential_9/lstm_1/while/lstm_cell_1/mul_9:z:0*
T0*'
_output_shapes
:??????????
6sequential_9/lstm_1/while/lstm_cell_1/ReadVariableOp_3ReadVariableOp?sequential_9_lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0?
;sequential_9/lstm_1/while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   ?
=sequential_9/lstm_1/while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
=sequential_9/lstm_1/while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
5sequential_9/lstm_1/while/lstm_cell_1/strided_slice_3StridedSlice>sequential_9/lstm_1/while/lstm_cell_1/ReadVariableOp_3:value:0Dsequential_9/lstm_1/while/lstm_cell_1/strided_slice_3/stack:output:0Fsequential_9/lstm_1/while/lstm_cell_1/strided_slice_3/stack_1:output:0Fsequential_9/lstm_1/while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
.sequential_9/lstm_1/while/lstm_cell_1/MatMul_7MatMul/sequential_9/lstm_1/while/lstm_cell_1/mul_7:z:0>sequential_9/lstm_1/while/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
+sequential_9/lstm_1/while/lstm_cell_1/add_4AddV28sequential_9/lstm_1/while/lstm_cell_1/BiasAdd_3:output:08sequential_9/lstm_1/while/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:??????????
/sequential_9/lstm_1/while/lstm_cell_1/Sigmoid_2Sigmoid/sequential_9/lstm_1/while/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:??????????
,sequential_9/lstm_1/while/lstm_cell_1/Tanh_1Tanh/sequential_9/lstm_1/while/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:??????????
,sequential_9/lstm_1/while/lstm_cell_1/mul_10Mul3sequential_9/lstm_1/while/lstm_cell_1/Sigmoid_2:y:00sequential_9/lstm_1/while/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:??????????
>sequential_9/lstm_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_9_lstm_1_while_placeholder_1%sequential_9_lstm_1_while_placeholder0sequential_9/lstm_1/while/lstm_cell_1/mul_10:z:0*
_output_shapes
: *
element_dtype0:???a
sequential_9/lstm_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
sequential_9/lstm_1/while/addAddV2%sequential_9_lstm_1_while_placeholder(sequential_9/lstm_1/while/add/y:output:0*
T0*
_output_shapes
: c
!sequential_9/lstm_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
sequential_9/lstm_1/while/add_1AddV2@sequential_9_lstm_1_while_sequential_9_lstm_1_while_loop_counter*sequential_9/lstm_1/while/add_1/y:output:0*
T0*
_output_shapes
: ?
"sequential_9/lstm_1/while/IdentityIdentity#sequential_9/lstm_1/while/add_1:z:0^sequential_9/lstm_1/while/NoOp*
T0*
_output_shapes
: ?
$sequential_9/lstm_1/while/Identity_1IdentityFsequential_9_lstm_1_while_sequential_9_lstm_1_while_maximum_iterations^sequential_9/lstm_1/while/NoOp*
T0*
_output_shapes
: ?
$sequential_9/lstm_1/while/Identity_2Identity!sequential_9/lstm_1/while/add:z:0^sequential_9/lstm_1/while/NoOp*
T0*
_output_shapes
: ?
$sequential_9/lstm_1/while/Identity_3IdentityNsequential_9/lstm_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_9/lstm_1/while/NoOp*
T0*
_output_shapes
: ?
$sequential_9/lstm_1/while/Identity_4Identity0sequential_9/lstm_1/while/lstm_cell_1/mul_10:z:0^sequential_9/lstm_1/while/NoOp*
T0*'
_output_shapes
:??????????
$sequential_9/lstm_1/while/Identity_5Identity/sequential_9/lstm_1/while/lstm_cell_1/add_3:z:0^sequential_9/lstm_1/while/NoOp*
T0*'
_output_shapes
:??????????
sequential_9/lstm_1/while/NoOpNoOp5^sequential_9/lstm_1/while/lstm_cell_1/ReadVariableOp7^sequential_9/lstm_1/while/lstm_cell_1/ReadVariableOp_17^sequential_9/lstm_1/while/lstm_cell_1/ReadVariableOp_27^sequential_9/lstm_1/while/lstm_cell_1/ReadVariableOp_3;^sequential_9/lstm_1/while/lstm_cell_1/split/ReadVariableOp=^sequential_9/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Q
"sequential_9_lstm_1_while_identity+sequential_9/lstm_1/while/Identity:output:0"U
$sequential_9_lstm_1_while_identity_1-sequential_9/lstm_1/while/Identity_1:output:0"U
$sequential_9_lstm_1_while_identity_2-sequential_9/lstm_1/while/Identity_2:output:0"U
$sequential_9_lstm_1_while_identity_3-sequential_9/lstm_1/while/Identity_3:output:0"U
$sequential_9_lstm_1_while_identity_4-sequential_9/lstm_1/while/Identity_4:output:0"U
$sequential_9_lstm_1_while_identity_5-sequential_9/lstm_1/while/Identity_5:output:0"?
=sequential_9_lstm_1_while_lstm_cell_1_readvariableop_resource?sequential_9_lstm_1_while_lstm_cell_1_readvariableop_resource_0"?
Esequential_9_lstm_1_while_lstm_cell_1_split_1_readvariableop_resourceGsequential_9_lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_0"?
Csequential_9_lstm_1_while_lstm_cell_1_split_readvariableop_resourceEsequential_9_lstm_1_while_lstm_cell_1_split_readvariableop_resource_0"?
=sequential_9_lstm_1_while_sequential_9_lstm_1_strided_slice_1?sequential_9_lstm_1_while_sequential_9_lstm_1_strided_slice_1_0"?
ysequential_9_lstm_1_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_1_tensorarrayunstack_tensorlistfromtensor{sequential_9_lstm_1_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????:?????????: : : : : 2l
4sequential_9/lstm_1/while/lstm_cell_1/ReadVariableOp4sequential_9/lstm_1/while/lstm_cell_1/ReadVariableOp2p
6sequential_9/lstm_1/while/lstm_cell_1/ReadVariableOp_16sequential_9/lstm_1/while/lstm_cell_1/ReadVariableOp_12p
6sequential_9/lstm_1/while/lstm_cell_1/ReadVariableOp_26sequential_9/lstm_1/while/lstm_cell_1/ReadVariableOp_22p
6sequential_9/lstm_1/while/lstm_cell_1/ReadVariableOp_36sequential_9/lstm_1/while/lstm_cell_1/ReadVariableOp_32x
:sequential_9/lstm_1/while/lstm_cell_1/split/ReadVariableOp:sequential_9/lstm_1/while/lstm_cell_1/split/ReadVariableOp2|
<sequential_9/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp<sequential_9/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
??
?
B__inference_lstm_1_layer_call_and_return_conditional_losses_163579
inputs_0;
)lstm_cell_1_split_readvariableop_resource: @9
+lstm_cell_1_split_1_readvariableop_resource:@5
#lstm_cell_1_readvariableop_resource:@
identity??lstm_cell_1/ReadVariableOp?lstm_cell_1/ReadVariableOp_1?lstm_cell_1/ReadVariableOp_2?lstm_cell_1/ReadVariableOp_3? lstm_cell_1/split/ReadVariableOp?"lstm_cell_1/split_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_maskc
lstm_cell_1/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:`
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? ^
lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/dropout/MulMullstm_cell_1/ones_like:output:0"lstm_cell_1/dropout/Const:output:0*
T0*'
_output_shapes
:????????? g
lstm_cell_1/dropout/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
0lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_1/dropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2??2g
"lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
 lstm_cell_1/dropout/GreaterEqualGreaterEqual9lstm_cell_1/dropout/random_uniform/RandomUniform:output:0+lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/dropout/CastCast$lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
lstm_cell_1/dropout/Mul_1Mullstm_cell_1/dropout/Mul:z:0lstm_cell_1/dropout/Cast:y:0*
T0*'
_output_shapes
:????????? `
lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/dropout_1/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_1/Const:output:0*
T0*'
_output_shapes
:????????? i
lstm_cell_1/dropout_1/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_1/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2?ʱi
$lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_1/dropout_1/GreaterEqualGreaterEqual;lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/dropout_1/CastCast&lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
lstm_cell_1/dropout_1/Mul_1Mullstm_cell_1/dropout_1/Mul:z:0lstm_cell_1/dropout_1/Cast:y:0*
T0*'
_output_shapes
:????????? `
lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/dropout_2/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_2/Const:output:0*
T0*'
_output_shapes
:????????? i
lstm_cell_1/dropout_2/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_2/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2??Bi
$lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_1/dropout_2/GreaterEqualGreaterEqual;lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/dropout_2/CastCast&lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
lstm_cell_1/dropout_2/Mul_1Mullstm_cell_1/dropout_2/Mul:z:0lstm_cell_1/dropout_2/Cast:y:0*
T0*'
_output_shapes
:????????? `
lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/dropout_3/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_3/Const:output:0*
T0*'
_output_shapes
:????????? i
lstm_cell_1/dropout_3/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_3/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2??i
$lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_1/dropout_3/GreaterEqualGreaterEqual;lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/dropout_3/CastCast&lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
lstm_cell_1/dropout_3/Mul_1Mullstm_cell_1/dropout_3/Mul:z:0lstm_cell_1/dropout_3/Cast:y:0*
T0*'
_output_shapes
:????????? [
lstm_cell_1/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:b
lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/ones_like_1Fill&lstm_cell_1/ones_like_1/Shape:output:0&lstm_cell_1/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????`
lstm_cell_1/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/dropout_4/MulMul lstm_cell_1/ones_like_1:output:0$lstm_cell_1/dropout_4/Const:output:0*
T0*'
_output_shapes
:?????????k
lstm_cell_1/dropout_4/ShapeShape lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_1/dropout_4/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2??*i
$lstm_cell_1/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_1/dropout_4/GreaterEqualGreaterEqual;lstm_cell_1/dropout_4/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/dropout_4/CastCast&lstm_cell_1/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
lstm_cell_1/dropout_4/Mul_1Mullstm_cell_1/dropout_4/Mul:z:0lstm_cell_1/dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????`
lstm_cell_1/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/dropout_5/MulMul lstm_cell_1/ones_like_1:output:0$lstm_cell_1/dropout_5/Const:output:0*
T0*'
_output_shapes
:?????????k
lstm_cell_1/dropout_5/ShapeShape lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_1/dropout_5/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2??ki
$lstm_cell_1/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_1/dropout_5/GreaterEqualGreaterEqual;lstm_cell_1/dropout_5/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/dropout_5/CastCast&lstm_cell_1/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
lstm_cell_1/dropout_5/Mul_1Mullstm_cell_1/dropout_5/Mul:z:0lstm_cell_1/dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????`
lstm_cell_1/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/dropout_6/MulMul lstm_cell_1/ones_like_1:output:0$lstm_cell_1/dropout_6/Const:output:0*
T0*'
_output_shapes
:?????????k
lstm_cell_1/dropout_6/ShapeShape lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_1/dropout_6/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_6/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2֙?i
$lstm_cell_1/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_1/dropout_6/GreaterEqualGreaterEqual;lstm_cell_1/dropout_6/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/dropout_6/CastCast&lstm_cell_1/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
lstm_cell_1/dropout_6/Mul_1Mullstm_cell_1/dropout_6/Mul:z:0lstm_cell_1/dropout_6/Cast:y:0*
T0*'
_output_shapes
:?????????`
lstm_cell_1/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/dropout_7/MulMul lstm_cell_1/ones_like_1:output:0$lstm_cell_1/dropout_7/Const:output:0*
T0*'
_output_shapes
:?????????k
lstm_cell_1/dropout_7/ShapeShape lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_1/dropout_7/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_7/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???i
$lstm_cell_1/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_1/dropout_7/GreaterEqualGreaterEqual;lstm_cell_1/dropout_7/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/dropout_7/CastCast&lstm_cell_1/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
lstm_cell_1/dropout_7/Mul_1Mullstm_cell_1/dropout_7/Mul:z:0lstm_cell_1/dropout_7/Cast:y:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/mulMulstrided_slice_2:output:0lstm_cell_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/mul_1Mulstrided_slice_2:output:0lstm_cell_1/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/mul_2Mulstrided_slice_2:output:0lstm_cell_1/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/mul_3Mulstrided_slice_2:output:0lstm_cell_1/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:????????? ]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_1/split/ReadVariableOpReadVariableOp)lstm_cell_1_split_readvariableop_resource*
_output_shapes

: @*
dtype0?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0(lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split
lstm_cell_1/MatMulMatMullstm_cell_1/mul:z:0lstm_cell_1/split:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/MatMul_1MatMullstm_cell_1/mul_1:z:0lstm_cell_1/split:output:1*
T0*'
_output_shapes
:??????????
lstm_cell_1/MatMul_2MatMullstm_cell_1/mul_2:z:0lstm_cell_1/split:output:2*
T0*'
_output_shapes
:??????????
lstm_cell_1/MatMul_3MatMullstm_cell_1/mul_3:z:0lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????_
lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_1/split_1/ReadVariableOpReadVariableOp+lstm_cell_1_split_1_readvariableop_resource*
_output_shapes
:@*
dtype0?
lstm_cell_1/split_1Split&lstm_cell_1/split_1/split_dim:output:0*lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/MatMul:product:0lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/BiasAdd_1BiasAddlstm_cell_1/MatMul_1:product:0lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:??????????
lstm_cell_1/BiasAdd_2BiasAddlstm_cell_1/MatMul_2:product:0lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
lstm_cell_1/BiasAdd_3BiasAddlstm_cell_1/MatMul_3:product:0lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????{
lstm_cell_1/mul_4Mulzeros:output:0lstm_cell_1/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:?????????{
lstm_cell_1/mul_5Mulzeros:output:0lstm_cell_1/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:?????????{
lstm_cell_1/mul_6Mulzeros:output:0lstm_cell_1/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:?????????{
lstm_cell_1/mul_7Mulzeros:output:0lstm_cell_1/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:?????????~
lstm_cell_1/ReadVariableOpReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0p
lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_1/strided_sliceStridedSlice"lstm_cell_1/ReadVariableOp:value:0(lstm_cell_1/strided_slice/stack:output:0*lstm_cell_1/strided_slice/stack_1:output:0*lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_1/MatMul_4MatMullstm_cell_1/mul_4:z:0"lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/addAddV2lstm_cell_1/BiasAdd:output:0lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????e
lstm_cell_1/SigmoidSigmoidlstm_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/ReadVariableOp_1ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0r
!lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_1/strided_slice_1StridedSlice$lstm_cell_1/ReadVariableOp_1:value:0*lstm_cell_1/strided_slice_1/stack:output:0,lstm_cell_1/strided_slice_1/stack_1:output:0,lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_1/MatMul_5MatMullstm_cell_1/mul_5:z:0$lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/add_1AddV2lstm_cell_1/BiasAdd_1:output:0lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????i
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????w
lstm_cell_1/mul_8Mullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/ReadVariableOp_2ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0r
!lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   t
#lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_1/strided_slice_2StridedSlice$lstm_cell_1/ReadVariableOp_2:value:0*lstm_cell_1/strided_slice_2/stack:output:0,lstm_cell_1/strided_slice_2/stack_1:output:0,lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_1/MatMul_6MatMullstm_cell_1/mul_6:z:0$lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/add_2AddV2lstm_cell_1/BiasAdd_2:output:0lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????a
lstm_cell_1/TanhTanhlstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????y
lstm_cell_1/mul_9Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????z
lstm_cell_1/add_3AddV2lstm_cell_1/mul_8:z:0lstm_cell_1/mul_9:z:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/ReadVariableOp_3ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0r
!lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   t
#lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_1/strided_slice_3StridedSlice$lstm_cell_1/ReadVariableOp_3:value:0*lstm_cell_1/strided_slice_3/stack:output:0,lstm_cell_1/strided_slice_3/stack_1:output:0,lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_1/MatMul_7MatMullstm_cell_1/mul_7:z:0$lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/add_4AddV2lstm_cell_1/BiasAdd_3:output:0lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????i
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????c
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????~
lstm_cell_1/mul_10Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_1_split_readvariableop_resource+lstm_cell_1_split_1_readvariableop_resource#lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_163381*
condR
while_cond_163380*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????????????? : : : 28
lstm_cell_1/ReadVariableOplstm_cell_1/ReadVariableOp2<
lstm_cell_1/ReadVariableOp_1lstm_cell_1/ReadVariableOp_12<
lstm_cell_1/ReadVariableOp_2lstm_cell_1/ReadVariableOp_22<
lstm_cell_1/ReadVariableOp_3lstm_cell_1/ReadVariableOp_32D
 lstm_cell_1/split/ReadVariableOp lstm_cell_1/split/ReadVariableOp2H
"lstm_cell_1/split_1/ReadVariableOp"lstm_cell_1/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :?????????????????? 
"
_user_specified_name
inputs/0
?
c
E__inference_dropout_9_layer_call_and_return_conditional_losses_164228

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????
[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
%sequential_9_lstm_1_while_cond_160592D
@sequential_9_lstm_1_while_sequential_9_lstm_1_while_loop_counterJ
Fsequential_9_lstm_1_while_sequential_9_lstm_1_while_maximum_iterations)
%sequential_9_lstm_1_while_placeholder+
'sequential_9_lstm_1_while_placeholder_1+
'sequential_9_lstm_1_while_placeholder_2+
'sequential_9_lstm_1_while_placeholder_3F
Bsequential_9_lstm_1_while_less_sequential_9_lstm_1_strided_slice_1\
Xsequential_9_lstm_1_while_sequential_9_lstm_1_while_cond_160592___redundant_placeholder0\
Xsequential_9_lstm_1_while_sequential_9_lstm_1_while_cond_160592___redundant_placeholder1\
Xsequential_9_lstm_1_while_sequential_9_lstm_1_while_cond_160592___redundant_placeholder2\
Xsequential_9_lstm_1_while_sequential_9_lstm_1_while_cond_160592___redundant_placeholder3&
"sequential_9_lstm_1_while_identity
?
sequential_9/lstm_1/while/LessLess%sequential_9_lstm_1_while_placeholderBsequential_9_lstm_1_while_less_sequential_9_lstm_1_strided_slice_1*
T0*
_output_shapes
: s
"sequential_9/lstm_1/while/IdentityIdentity"sequential_9/lstm_1/while/Less:z:0*
T0
*
_output_shapes
: "Q
"sequential_9_lstm_1_while_identity+sequential_9/lstm_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_161569

inputs%
embedding_9_161274:	?' 
lstm_1_161520: @
lstm_1_161522:@
lstm_1_161524:@!
dense_18_161539:

dense_18_161541:
!
dense_19_161563:

dense_19_161565:
identity?? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall?#embedding_9/StatefulPartitionedCall?lstm_1/StatefulPartitionedCall?
#embedding_9/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_9_161274*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_9_layer_call_and_return_conditional_losses_161273?
lstm_1/StatefulPartitionedCallStatefulPartitionedCall,embedding_9/StatefulPartitionedCall:output:0lstm_1_161520lstm_1_161522lstm_1_161524*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_161519?
 dense_18/StatefulPartitionedCallStatefulPartitionedCall'lstm_1/StatefulPartitionedCall:output:0dense_18_161539dense_18_161541*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_161538?
dropout_9/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_161549?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0dense_19_161563dense_19_161565*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_161562x
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall$^embedding_9/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????????????: : : : : : : : 2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2J
#embedding_9/StatefulPartitionedCall#embedding_9/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?	
d
E__inference_dropout_9_layer_call_and_return_conditional_losses_164240

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????
o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????
Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?	
?
-__inference_sequential_9_layer_call_fn_162220

inputs
unknown:	?' 
	unknown_0: @
	unknown_1:@
	unknown_2:@
	unknown_3:

	unknown_4:

	unknown_5:

	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_161569o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
??
?
B__inference_lstm_1_layer_call_and_return_conditional_losses_164193

inputs;
)lstm_cell_1_split_readvariableop_resource: @9
+lstm_cell_1_split_1_readvariableop_resource:@5
#lstm_cell_1_readvariableop_resource:@
identity??lstm_cell_1/ReadVariableOp?lstm_cell_1/ReadVariableOp_1?lstm_cell_1/ReadVariableOp_2?lstm_cell_1/ReadVariableOp_3? lstm_cell_1/split/ReadVariableOp?"lstm_cell_1/split_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_maskc
lstm_cell_1/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:`
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? ^
lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/dropout/MulMullstm_cell_1/ones_like:output:0"lstm_cell_1/dropout/Const:output:0*
T0*'
_output_shapes
:????????? g
lstm_cell_1/dropout/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
0lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_1/dropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???g
"lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
 lstm_cell_1/dropout/GreaterEqualGreaterEqual9lstm_cell_1/dropout/random_uniform/RandomUniform:output:0+lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/dropout/CastCast$lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
lstm_cell_1/dropout/Mul_1Mullstm_cell_1/dropout/Mul:z:0lstm_cell_1/dropout/Cast:y:0*
T0*'
_output_shapes
:????????? `
lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/dropout_1/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_1/Const:output:0*
T0*'
_output_shapes
:????????? i
lstm_cell_1/dropout_1/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_1/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???i
$lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_1/dropout_1/GreaterEqualGreaterEqual;lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/dropout_1/CastCast&lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
lstm_cell_1/dropout_1/Mul_1Mullstm_cell_1/dropout_1/Mul:z:0lstm_cell_1/dropout_1/Cast:y:0*
T0*'
_output_shapes
:????????? `
lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/dropout_2/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_2/Const:output:0*
T0*'
_output_shapes
:????????? i
lstm_cell_1/dropout_2/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_2/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???i
$lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_1/dropout_2/GreaterEqualGreaterEqual;lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/dropout_2/CastCast&lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
lstm_cell_1/dropout_2/Mul_1Mullstm_cell_1/dropout_2/Mul:z:0lstm_cell_1/dropout_2/Cast:y:0*
T0*'
_output_shapes
:????????? `
lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/dropout_3/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_3/Const:output:0*
T0*'
_output_shapes
:????????? i
lstm_cell_1/dropout_3/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_3/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???i
$lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_1/dropout_3/GreaterEqualGreaterEqual;lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/dropout_3/CastCast&lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
lstm_cell_1/dropout_3/Mul_1Mullstm_cell_1/dropout_3/Mul:z:0lstm_cell_1/dropout_3/Cast:y:0*
T0*'
_output_shapes
:????????? [
lstm_cell_1/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:b
lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/ones_like_1Fill&lstm_cell_1/ones_like_1/Shape:output:0&lstm_cell_1/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????`
lstm_cell_1/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/dropout_4/MulMul lstm_cell_1/ones_like_1:output:0$lstm_cell_1/dropout_4/Const:output:0*
T0*'
_output_shapes
:?????????k
lstm_cell_1/dropout_4/ShapeShape lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_1/dropout_4/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???i
$lstm_cell_1/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_1/dropout_4/GreaterEqualGreaterEqual;lstm_cell_1/dropout_4/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/dropout_4/CastCast&lstm_cell_1/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
lstm_cell_1/dropout_4/Mul_1Mullstm_cell_1/dropout_4/Mul:z:0lstm_cell_1/dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????`
lstm_cell_1/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/dropout_5/MulMul lstm_cell_1/ones_like_1:output:0$lstm_cell_1/dropout_5/Const:output:0*
T0*'
_output_shapes
:?????????k
lstm_cell_1/dropout_5/ShapeShape lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_1/dropout_5/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2Ȏ?i
$lstm_cell_1/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_1/dropout_5/GreaterEqualGreaterEqual;lstm_cell_1/dropout_5/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/dropout_5/CastCast&lstm_cell_1/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
lstm_cell_1/dropout_5/Mul_1Mullstm_cell_1/dropout_5/Mul:z:0lstm_cell_1/dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????`
lstm_cell_1/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/dropout_6/MulMul lstm_cell_1/ones_like_1:output:0$lstm_cell_1/dropout_6/Const:output:0*
T0*'
_output_shapes
:?????????k
lstm_cell_1/dropout_6/ShapeShape lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_1/dropout_6/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_6/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2?نi
$lstm_cell_1/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_1/dropout_6/GreaterEqualGreaterEqual;lstm_cell_1/dropout_6/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/dropout_6/CastCast&lstm_cell_1/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
lstm_cell_1/dropout_6/Mul_1Mullstm_cell_1/dropout_6/Mul:z:0lstm_cell_1/dropout_6/Cast:y:0*
T0*'
_output_shapes
:?????????`
lstm_cell_1/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_1/dropout_7/MulMul lstm_cell_1/ones_like_1:output:0$lstm_cell_1/dropout_7/Const:output:0*
T0*'
_output_shapes
:?????????k
lstm_cell_1/dropout_7/ShapeShape lstm_cell_1/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_1/dropout_7/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_7/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???i
$lstm_cell_1/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_1/dropout_7/GreaterEqualGreaterEqual;lstm_cell_1/dropout_7/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/dropout_7/CastCast&lstm_cell_1/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
lstm_cell_1/dropout_7/Mul_1Mullstm_cell_1/dropout_7/Mul:z:0lstm_cell_1/dropout_7/Cast:y:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/mulMulstrided_slice_2:output:0lstm_cell_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/mul_1Mulstrided_slice_2:output:0lstm_cell_1/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/mul_2Mulstrided_slice_2:output:0lstm_cell_1/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
lstm_cell_1/mul_3Mulstrided_slice_2:output:0lstm_cell_1/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:????????? ]
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_1/split/ReadVariableOpReadVariableOp)lstm_cell_1_split_readvariableop_resource*
_output_shapes

: @*
dtype0?
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0(lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split
lstm_cell_1/MatMulMatMullstm_cell_1/mul:z:0lstm_cell_1/split:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/MatMul_1MatMullstm_cell_1/mul_1:z:0lstm_cell_1/split:output:1*
T0*'
_output_shapes
:??????????
lstm_cell_1/MatMul_2MatMullstm_cell_1/mul_2:z:0lstm_cell_1/split:output:2*
T0*'
_output_shapes
:??????????
lstm_cell_1/MatMul_3MatMullstm_cell_1/mul_3:z:0lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????_
lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_1/split_1/ReadVariableOpReadVariableOp+lstm_cell_1_split_1_readvariableop_resource*
_output_shapes
:@*
dtype0?
lstm_cell_1/split_1Split&lstm_cell_1/split_1/split_dim:output:0*lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
lstm_cell_1/BiasAddBiasAddlstm_cell_1/MatMul:product:0lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/BiasAdd_1BiasAddlstm_cell_1/MatMul_1:product:0lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:??????????
lstm_cell_1/BiasAdd_2BiasAddlstm_cell_1/MatMul_2:product:0lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
lstm_cell_1/BiasAdd_3BiasAddlstm_cell_1/MatMul_3:product:0lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:?????????{
lstm_cell_1/mul_4Mulzeros:output:0lstm_cell_1/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:?????????{
lstm_cell_1/mul_5Mulzeros:output:0lstm_cell_1/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:?????????{
lstm_cell_1/mul_6Mulzeros:output:0lstm_cell_1/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:?????????{
lstm_cell_1/mul_7Mulzeros:output:0lstm_cell_1/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:?????????~
lstm_cell_1/ReadVariableOpReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0p
lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_1/strided_sliceStridedSlice"lstm_cell_1/ReadVariableOp:value:0(lstm_cell_1/strided_slice/stack:output:0*lstm_cell_1/strided_slice/stack_1:output:0*lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_1/MatMul_4MatMullstm_cell_1/mul_4:z:0"lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/addAddV2lstm_cell_1/BiasAdd:output:0lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????e
lstm_cell_1/SigmoidSigmoidlstm_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/ReadVariableOp_1ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0r
!lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_1/strided_slice_1StridedSlice$lstm_cell_1/ReadVariableOp_1:value:0*lstm_cell_1/strided_slice_1/stack:output:0,lstm_cell_1/strided_slice_1/stack_1:output:0,lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_1/MatMul_5MatMullstm_cell_1/mul_5:z:0$lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/add_1AddV2lstm_cell_1/BiasAdd_1:output:0lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????i
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:?????????w
lstm_cell_1/mul_8Mullstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/ReadVariableOp_2ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0r
!lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   t
#lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_1/strided_slice_2StridedSlice$lstm_cell_1/ReadVariableOp_2:value:0*lstm_cell_1/strided_slice_2/stack:output:0,lstm_cell_1/strided_slice_2/stack_1:output:0,lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_1/MatMul_6MatMullstm_cell_1/mul_6:z:0$lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/add_2AddV2lstm_cell_1/BiasAdd_2:output:0lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????a
lstm_cell_1/TanhTanhlstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:?????????y
lstm_cell_1/mul_9Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:?????????z
lstm_cell_1/add_3AddV2lstm_cell_1/mul_8:z:0lstm_cell_1/mul_9:z:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/ReadVariableOp_3ReadVariableOp#lstm_cell_1_readvariableop_resource*
_output_shapes

:@*
dtype0r
!lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   t
#lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_1/strided_slice_3StridedSlice$lstm_cell_1/ReadVariableOp_3:value:0*lstm_cell_1/strided_slice_3/stack:output:0,lstm_cell_1/strided_slice_3/stack_1:output:0,lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
lstm_cell_1/MatMul_7MatMullstm_cell_1/mul_7:z:0$lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
lstm_cell_1/add_4AddV2lstm_cell_1/BiasAdd_3:output:0lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????i
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????c
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:?????????~
lstm_cell_1/mul_10Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_1_split_readvariableop_resource+lstm_cell_1_split_1_readvariableop_resource#lstm_cell_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_163995*
condR
while_cond_163994*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^lstm_cell_1/ReadVariableOp^lstm_cell_1/ReadVariableOp_1^lstm_cell_1/ReadVariableOp_2^lstm_cell_1/ReadVariableOp_3!^lstm_cell_1/split/ReadVariableOp#^lstm_cell_1/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????????????? : : : 28
lstm_cell_1/ReadVariableOplstm_cell_1/ReadVariableOp2<
lstm_cell_1/ReadVariableOp_1lstm_cell_1/ReadVariableOp_12<
lstm_cell_1/ReadVariableOp_2lstm_cell_1/ReadVariableOp_22<
lstm_cell_1/ReadVariableOp_3lstm_cell_1/ReadVariableOp_32D
 lstm_cell_1/split/ReadVariableOp lstm_cell_1/split/ReadVariableOp2H
"lstm_cell_1/split_1/ReadVariableOp"lstm_cell_1/split_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
??
?
lstm_1_while_body_162356*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3)
%lstm_1_while_lstm_1_strided_slice_1_0e
alstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0J
8lstm_1_while_lstm_cell_1_split_readvariableop_resource_0: @H
:lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_0:@D
2lstm_1_while_lstm_cell_1_readvariableop_resource_0:@
lstm_1_while_identity
lstm_1_while_identity_1
lstm_1_while_identity_2
lstm_1_while_identity_3
lstm_1_while_identity_4
lstm_1_while_identity_5'
#lstm_1_while_lstm_1_strided_slice_1c
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensorH
6lstm_1_while_lstm_cell_1_split_readvariableop_resource: @F
8lstm_1_while_lstm_cell_1_split_1_readvariableop_resource:@B
0lstm_1_while_lstm_cell_1_readvariableop_resource:@??'lstm_1/while/lstm_cell_1/ReadVariableOp?)lstm_1/while/lstm_cell_1/ReadVariableOp_1?)lstm_1/while/lstm_cell_1/ReadVariableOp_2?)lstm_1/while/lstm_cell_1/ReadVariableOp_3?-lstm_1/while/lstm_cell_1/split/ReadVariableOp?/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp?
>lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
0lstm_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0lstm_1_while_placeholderGlstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype0?
(lstm_1/while/lstm_cell_1/ones_like/ShapeShape7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:m
(lstm_1/while/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
"lstm_1/while/lstm_cell_1/ones_likeFill1lstm_1/while/lstm_cell_1/ones_like/Shape:output:01lstm_1/while/lstm_cell_1/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? t
*lstm_1/while/lstm_cell_1/ones_like_1/ShapeShapelstm_1_while_placeholder_2*
T0*
_output_shapes
:o
*lstm_1/while/lstm_cell_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
$lstm_1/while/lstm_cell_1/ones_like_1Fill3lstm_1/while/lstm_cell_1/ones_like_1/Shape:output:03lstm_1/while/lstm_cell_1/ones_like_1/Const:output:0*
T0*'
_output_shapes
:??????????
lstm_1/while/lstm_cell_1/mulMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
lstm_1/while/lstm_cell_1/mul_1Mul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
lstm_1/while/lstm_cell_1/mul_2Mul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
lstm_1/while/lstm_cell_1/mul_3Mul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_1/while/lstm_cell_1/ones_like:output:0*
T0*'
_output_shapes
:????????? j
(lstm_1/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
-lstm_1/while/lstm_cell_1/split/ReadVariableOpReadVariableOp8lstm_1_while_lstm_cell_1_split_readvariableop_resource_0*
_output_shapes

: @*
dtype0?
lstm_1/while/lstm_cell_1/splitSplit1lstm_1/while/lstm_cell_1/split/split_dim:output:05lstm_1/while/lstm_cell_1/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: : : : *
	num_split?
lstm_1/while/lstm_cell_1/MatMulMatMul lstm_1/while/lstm_cell_1/mul:z:0'lstm_1/while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:??????????
!lstm_1/while/lstm_cell_1/MatMul_1MatMul"lstm_1/while/lstm_cell_1/mul_1:z:0'lstm_1/while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:??????????
!lstm_1/while/lstm_cell_1/MatMul_2MatMul"lstm_1/while/lstm_cell_1/mul_2:z:0'lstm_1/while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:??????????
!lstm_1/while/lstm_cell_1/MatMul_3MatMul"lstm_1/while/lstm_cell_1/mul_3:z:0'lstm_1/while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:?????????l
*lstm_1/while/lstm_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
/lstm_1/while/lstm_cell_1/split_1/ReadVariableOpReadVariableOp:lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_0*
_output_shapes
:@*
dtype0?
 lstm_1/while/lstm_cell_1/split_1Split3lstm_1/while/lstm_cell_1/split_1/split_dim:output:07lstm_1/while/lstm_cell_1/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split?
 lstm_1/while/lstm_cell_1/BiasAddBiasAdd)lstm_1/while/lstm_cell_1/MatMul:product:0)lstm_1/while/lstm_cell_1/split_1:output:0*
T0*'
_output_shapes
:??????????
"lstm_1/while/lstm_cell_1/BiasAdd_1BiasAdd+lstm_1/while/lstm_cell_1/MatMul_1:product:0)lstm_1/while/lstm_cell_1/split_1:output:1*
T0*'
_output_shapes
:??????????
"lstm_1/while/lstm_cell_1/BiasAdd_2BiasAdd+lstm_1/while/lstm_cell_1/MatMul_2:product:0)lstm_1/while/lstm_cell_1/split_1:output:2*
T0*'
_output_shapes
:??????????
"lstm_1/while/lstm_cell_1/BiasAdd_3BiasAdd+lstm_1/while/lstm_cell_1/MatMul_3:product:0)lstm_1/while/lstm_cell_1/split_1:output:3*
T0*'
_output_shapes
:??????????
lstm_1/while/lstm_cell_1/mul_4Mullstm_1_while_placeholder_2-lstm_1/while/lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
lstm_1/while/lstm_cell_1/mul_5Mullstm_1_while_placeholder_2-lstm_1/while/lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
lstm_1/while/lstm_cell_1/mul_6Mullstm_1_while_placeholder_2-lstm_1/while/lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
lstm_1/while/lstm_cell_1/mul_7Mullstm_1_while_placeholder_2-lstm_1/while/lstm_cell_1/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
'lstm_1/while/lstm_cell_1/ReadVariableOpReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0}
,lstm_1/while/lstm_cell_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.lstm_1/while/lstm_cell_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.lstm_1/while/lstm_cell_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
&lstm_1/while/lstm_cell_1/strided_sliceStridedSlice/lstm_1/while/lstm_cell_1/ReadVariableOp:value:05lstm_1/while/lstm_cell_1/strided_slice/stack:output:07lstm_1/while/lstm_cell_1/strided_slice/stack_1:output:07lstm_1/while/lstm_cell_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
!lstm_1/while/lstm_cell_1/MatMul_4MatMul"lstm_1/while/lstm_cell_1/mul_4:z:0/lstm_1/while/lstm_cell_1/strided_slice:output:0*
T0*'
_output_shapes
:??????????
lstm_1/while/lstm_cell_1/addAddV2)lstm_1/while/lstm_cell_1/BiasAdd:output:0+lstm_1/while/lstm_cell_1/MatMul_4:product:0*
T0*'
_output_shapes
:?????????
 lstm_1/while/lstm_cell_1/SigmoidSigmoid lstm_1/while/lstm_cell_1/add:z:0*
T0*'
_output_shapes
:??????????
)lstm_1/while/lstm_cell_1/ReadVariableOp_1ReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0
.lstm_1/while/lstm_cell_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       ?
0lstm_1/while/lstm_cell_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
0lstm_1/while/lstm_cell_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_1/while/lstm_cell_1/strided_slice_1StridedSlice1lstm_1/while/lstm_cell_1/ReadVariableOp_1:value:07lstm_1/while/lstm_cell_1/strided_slice_1/stack:output:09lstm_1/while/lstm_cell_1/strided_slice_1/stack_1:output:09lstm_1/while/lstm_cell_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
!lstm_1/while/lstm_cell_1/MatMul_5MatMul"lstm_1/while/lstm_cell_1/mul_5:z:01lstm_1/while/lstm_cell_1/strided_slice_1:output:0*
T0*'
_output_shapes
:??????????
lstm_1/while/lstm_cell_1/add_1AddV2+lstm_1/while/lstm_cell_1/BiasAdd_1:output:0+lstm_1/while/lstm_cell_1/MatMul_5:product:0*
T0*'
_output_shapes
:??????????
"lstm_1/while/lstm_cell_1/Sigmoid_1Sigmoid"lstm_1/while/lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:??????????
lstm_1/while/lstm_cell_1/mul_8Mul&lstm_1/while/lstm_cell_1/Sigmoid_1:y:0lstm_1_while_placeholder_3*
T0*'
_output_shapes
:??????????
)lstm_1/while/lstm_cell_1/ReadVariableOp_2ReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0
.lstm_1/while/lstm_cell_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
0lstm_1/while/lstm_cell_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    0   ?
0lstm_1/while/lstm_cell_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_1/while/lstm_cell_1/strided_slice_2StridedSlice1lstm_1/while/lstm_cell_1/ReadVariableOp_2:value:07lstm_1/while/lstm_cell_1/strided_slice_2/stack:output:09lstm_1/while/lstm_cell_1/strided_slice_2/stack_1:output:09lstm_1/while/lstm_cell_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
!lstm_1/while/lstm_cell_1/MatMul_6MatMul"lstm_1/while/lstm_cell_1/mul_6:z:01lstm_1/while/lstm_cell_1/strided_slice_2:output:0*
T0*'
_output_shapes
:??????????
lstm_1/while/lstm_cell_1/add_2AddV2+lstm_1/while/lstm_cell_1/BiasAdd_2:output:0+lstm_1/while/lstm_cell_1/MatMul_6:product:0*
T0*'
_output_shapes
:?????????{
lstm_1/while/lstm_cell_1/TanhTanh"lstm_1/while/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:??????????
lstm_1/while/lstm_cell_1/mul_9Mul$lstm_1/while/lstm_cell_1/Sigmoid:y:0!lstm_1/while/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:??????????
lstm_1/while/lstm_cell_1/add_3AddV2"lstm_1/while/lstm_cell_1/mul_8:z:0"lstm_1/while/lstm_cell_1/mul_9:z:0*
T0*'
_output_shapes
:??????????
)lstm_1/while/lstm_cell_1/ReadVariableOp_3ReadVariableOp2lstm_1_while_lstm_cell_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0
.lstm_1/while/lstm_cell_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    0   ?
0lstm_1/while/lstm_cell_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
0lstm_1/while/lstm_cell_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_1/while/lstm_cell_1/strided_slice_3StridedSlice1lstm_1/while/lstm_cell_1/ReadVariableOp_3:value:07lstm_1/while/lstm_cell_1/strided_slice_3/stack:output:09lstm_1/while/lstm_cell_1/strided_slice_3/stack_1:output:09lstm_1/while/lstm_cell_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask?
!lstm_1/while/lstm_cell_1/MatMul_7MatMul"lstm_1/while/lstm_cell_1/mul_7:z:01lstm_1/while/lstm_cell_1/strided_slice_3:output:0*
T0*'
_output_shapes
:??????????
lstm_1/while/lstm_cell_1/add_4AddV2+lstm_1/while/lstm_cell_1/BiasAdd_3:output:0+lstm_1/while/lstm_cell_1/MatMul_7:product:0*
T0*'
_output_shapes
:??????????
"lstm_1/while/lstm_cell_1/Sigmoid_2Sigmoid"lstm_1/while/lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:?????????}
lstm_1/while/lstm_cell_1/Tanh_1Tanh"lstm_1/while/lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:??????????
lstm_1/while/lstm_cell_1/mul_10Mul&lstm_1/while/lstm_cell_1/Sigmoid_2:y:0#lstm_1/while/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:??????????
1lstm_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_1_while_placeholder_1lstm_1_while_placeholder#lstm_1/while/lstm_cell_1/mul_10:z:0*
_output_shapes
: *
element_dtype0:???T
lstm_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_1/while/addAddV2lstm_1_while_placeholderlstm_1/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_1/while/add_1AddV2&lstm_1_while_lstm_1_while_loop_counterlstm_1/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_1/while/IdentityIdentitylstm_1/while/add_1:z:0^lstm_1/while/NoOp*
T0*
_output_shapes
: ?
lstm_1/while/Identity_1Identity,lstm_1_while_lstm_1_while_maximum_iterations^lstm_1/while/NoOp*
T0*
_output_shapes
: n
lstm_1/while/Identity_2Identitylstm_1/while/add:z:0^lstm_1/while/NoOp*
T0*
_output_shapes
: ?
lstm_1/while/Identity_3IdentityAlstm_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_1/while/NoOp*
T0*
_output_shapes
: ?
lstm_1/while/Identity_4Identity#lstm_1/while/lstm_cell_1/mul_10:z:0^lstm_1/while/NoOp*
T0*'
_output_shapes
:??????????
lstm_1/while/Identity_5Identity"lstm_1/while/lstm_cell_1/add_3:z:0^lstm_1/while/NoOp*
T0*'
_output_shapes
:??????????
lstm_1/while/NoOpNoOp(^lstm_1/while/lstm_cell_1/ReadVariableOp*^lstm_1/while/lstm_cell_1/ReadVariableOp_1*^lstm_1/while/lstm_cell_1/ReadVariableOp_2*^lstm_1/while/lstm_cell_1/ReadVariableOp_3.^lstm_1/while/lstm_cell_1/split/ReadVariableOp0^lstm_1/while/lstm_cell_1/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_1_while_identitylstm_1/while/Identity:output:0";
lstm_1_while_identity_1 lstm_1/while/Identity_1:output:0";
lstm_1_while_identity_2 lstm_1/while/Identity_2:output:0";
lstm_1_while_identity_3 lstm_1/while/Identity_3:output:0";
lstm_1_while_identity_4 lstm_1/while/Identity_4:output:0";
lstm_1_while_identity_5 lstm_1/while/Identity_5:output:0"L
#lstm_1_while_lstm_1_strided_slice_1%lstm_1_while_lstm_1_strided_slice_1_0"f
0lstm_1_while_lstm_cell_1_readvariableop_resource2lstm_1_while_lstm_cell_1_readvariableop_resource_0"v
8lstm_1_while_lstm_cell_1_split_1_readvariableop_resource:lstm_1_while_lstm_cell_1_split_1_readvariableop_resource_0"r
6lstm_1_while_lstm_cell_1_split_readvariableop_resource8lstm_1_while_lstm_cell_1_split_readvariableop_resource_0"?
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensoralstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????:?????????: : : : : 2R
'lstm_1/while/lstm_cell_1/ReadVariableOp'lstm_1/while/lstm_cell_1/ReadVariableOp2V
)lstm_1/while/lstm_cell_1/ReadVariableOp_1)lstm_1/while/lstm_cell_1/ReadVariableOp_12V
)lstm_1/while/lstm_cell_1/ReadVariableOp_2)lstm_1/while/lstm_cell_1/ReadVariableOp_22V
)lstm_1/while/lstm_cell_1/ReadVariableOp_3)lstm_1/while/lstm_cell_1/ReadVariableOp_32^
-lstm_1/while/lstm_cell_1/split/ReadVariableOp-lstm_1/while/lstm_cell_1/split/ReadVariableOp2b
/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp/lstm_1/while/lstm_cell_1/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?7
?
B__inference_lstm_1_layer_call_and_return_conditional_losses_160942

inputs$
lstm_cell_1_160860: @ 
lstm_cell_1_160862:@$
lstm_cell_1_160864:@
identity??#lstm_cell_1/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask?
#lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_1_160860lstm_cell_1_160862lstm_cell_1_160864*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_160859n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_1_160860lstm_cell_1_160862lstm_cell_1_160864*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_160873*
condR
while_cond_160872*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????t
NoOpNoOp$^lstm_cell_1/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????????????? : : : 2J
#lstm_cell_1/StatefulPartitionedCall#lstm_cell_1/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?
?
'__inference_lstm_1_layer_call_fn_162943
inputs_0
unknown: @
	unknown_0:@
	unknown_1:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_lstm_1_layer_call_and_return_conditional_losses_161247o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????????????? : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :?????????????????? 
"
_user_specified_name
inputs/0"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
X
embedding_9_inputC
#serving_default_embedding_9_input:0??????????????????<
dense_190
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer_with_weights-3
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
s__call__
*t&call_and_return_all_conditional_losses
u_default_save_signature"
_tf_keras_sequential
?

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
v__call__
*w&call_and_return_all_conditional_losses"
_tf_keras_layer
?
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
x__call__
*y&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
 	keras_api
|__call__
*}&call_and_return_all_conditional_losses"
_tf_keras_layer
?

!kernel
"bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?
'iter

(beta_1

)beta_2
	*decay
+learning_ratemcmdme!mf"mg,mh-mi.mjvkvlvm!vn"vo,vp-vq.vr"
	optimizer
X
0
,1
-2
.3
4
5
!6
"7"
trackable_list_wrapper
X
0
,1
-2
.3
4
5
!6
"7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
/non_trainable_variables

0layers
1metrics
2layer_regularization_losses
3layer_metrics
	variables
trainable_variables
	regularization_losses
s__call__
u_default_save_signature
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
):'	?' 2embedding_9/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
?
9
state_size

,kernel
-recurrent_kernel
.bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
,0
-1
.2"
trackable_list_wrapper
5
,0
-1
.2"
trackable_list_wrapper
 "
trackable_list_wrapper
?

>states
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
!:
2dense_18/kernel
:
2dense_18/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
	variables
trainable_variables
regularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
regularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
!:
2dense_19/kernel
:2dense_19/bias
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
#	variables
$trainable_variables
%regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
+:) @2lstm_1/lstm_cell_1/kernel
5:3@2#lstm_1/lstm_cell_1/recurrent_kernel
%:#@2lstm_1/lstm_cell_1/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
,0
-1
.2"
trackable_list_wrapper
5
,0
-1
.2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
:	variables
;trainable_variables
<regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Ztotal
	[count
\	variables
]	keras_api"
_tf_keras_metric
^
	^total
	_count
`
_fn_kwargs
a	variables
b	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
Z0
[1"
trackable_list_wrapper
-
\	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
^0
_1"
trackable_list_wrapper
-
a	variables"
_generic_user_object
.:,	?' 2Adam/embedding_9/embeddings/m
&:$
2Adam/dense_18/kernel/m
 :
2Adam/dense_18/bias/m
&:$
2Adam/dense_19/kernel/m
 :2Adam/dense_19/bias/m
0:. @2 Adam/lstm_1/lstm_cell_1/kernel/m
::8@2*Adam/lstm_1/lstm_cell_1/recurrent_kernel/m
*:(@2Adam/lstm_1/lstm_cell_1/bias/m
.:,	?' 2Adam/embedding_9/embeddings/v
&:$
2Adam/dense_18/kernel/v
 :
2Adam/dense_18/bias/v
&:$
2Adam/dense_19/kernel/v
 :2Adam/dense_19/bias/v
0:. @2 Adam/lstm_1/lstm_cell_1/kernel/v
::8@2*Adam/lstm_1/lstm_cell_1/recurrent_kernel/v
*:(@2Adam/lstm_1/lstm_cell_1/bias/v
?2?
-__inference_sequential_9_layer_call_fn_161588
-__inference_sequential_9_layer_call_fn_162220
-__inference_sequential_9_layer_call_fn_162241
-__inference_sequential_9_layer_call_fn_162120?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_sequential_9_layer_call_and_return_conditional_losses_162505
H__inference_sequential_9_layer_call_and_return_conditional_losses_162904
H__inference_sequential_9_layer_call_and_return_conditional_losses_162145
H__inference_sequential_9_layer_call_and_return_conditional_losses_162170?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_160742embedding_9_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_9_layer_call_fn_162911?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_9_layer_call_and_return_conditional_losses_162921?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_lstm_1_layer_call_fn_162932
'__inference_lstm_1_layer_call_fn_162943
'__inference_lstm_1_layer_call_fn_162954
'__inference_lstm_1_layer_call_fn_162965?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_lstm_1_layer_call_and_return_conditional_losses_163208
B__inference_lstm_1_layer_call_and_return_conditional_losses_163579
B__inference_lstm_1_layer_call_and_return_conditional_losses_163822
B__inference_lstm_1_layer_call_and_return_conditional_losses_164193?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dense_18_layer_call_fn_164202?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_18_layer_call_and_return_conditional_losses_164213?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dropout_9_layer_call_fn_164218
*__inference_dropout_9_layer_call_fn_164223?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_9_layer_call_and_return_conditional_losses_164228
E__inference_dropout_9_layer_call_and_return_conditional_losses_164240?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dense_19_layer_call_fn_164249?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_19_layer_call_and_return_conditional_losses_164260?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_162199embedding_9_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_lstm_cell_1_layer_call_fn_164277
,__inference_lstm_cell_1_layer_call_fn_164294?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_164376
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_164522?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
!__inference__wrapped_model_160742?,.-!"C?@
9?6
4?1
embedding_9_input??????????????????
? "3?0
.
dense_19"?
dense_19??????????
D__inference_dense_18_layer_call_and_return_conditional_losses_164213\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????

? |
)__inference_dense_18_layer_call_fn_164202O/?,
%?"
 ?
inputs?????????
? "??????????
?
D__inference_dense_19_layer_call_and_return_conditional_losses_164260\!"/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????
? |
)__inference_dense_19_layer_call_fn_164249O!"/?,
%?"
 ?
inputs?????????

? "???????????
E__inference_dropout_9_layer_call_and_return_conditional_losses_164228\3?0
)?&
 ?
inputs?????????

p 
? "%?"
?
0?????????

? ?
E__inference_dropout_9_layer_call_and_return_conditional_losses_164240\3?0
)?&
 ?
inputs?????????

p
? "%?"
?
0?????????

? }
*__inference_dropout_9_layer_call_fn_164218O3?0
)?&
 ?
inputs?????????

p 
? "??????????
}
*__inference_dropout_9_layer_call_fn_164223O3?0
)?&
 ?
inputs?????????

p
? "??????????
?
G__inference_embedding_9_layer_call_and_return_conditional_losses_162921q8?5
.?+
)?&
inputs??????????????????
? "2?/
(?%
0?????????????????? 
? ?
,__inference_embedding_9_layer_call_fn_162911d8?5
.?+
)?&
inputs??????????????????
? "%?"?????????????????? ?
B__inference_lstm_1_layer_call_and_return_conditional_losses_163208},.-O?L
E?B
4?1
/?,
inputs/0?????????????????? 

 
p 

 
? "%?"
?
0?????????
? ?
B__inference_lstm_1_layer_call_and_return_conditional_losses_163579},.-O?L
E?B
4?1
/?,
inputs/0?????????????????? 

 
p

 
? "%?"
?
0?????????
? ?
B__inference_lstm_1_layer_call_and_return_conditional_losses_163822v,.-H?E
>?;
-?*
inputs?????????????????? 

 
p 

 
? "%?"
?
0?????????
? ?
B__inference_lstm_1_layer_call_and_return_conditional_losses_164193v,.-H?E
>?;
-?*
inputs?????????????????? 

 
p

 
? "%?"
?
0?????????
? ?
'__inference_lstm_1_layer_call_fn_162932p,.-O?L
E?B
4?1
/?,
inputs/0?????????????????? 

 
p 

 
? "???????????
'__inference_lstm_1_layer_call_fn_162943p,.-O?L
E?B
4?1
/?,
inputs/0?????????????????? 

 
p

 
? "???????????
'__inference_lstm_1_layer_call_fn_162954i,.-H?E
>?;
-?*
inputs?????????????????? 

 
p 

 
? "???????????
'__inference_lstm_1_layer_call_fn_162965i,.-H?E
>?;
-?*
inputs?????????????????? 

 
p

 
? "???????????
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_164376?,.-??}
v?s
 ?
inputs????????? 
K?H
"?
states/0?????????
"?
states/1?????????
p 
? "s?p
i?f
?
0/0?????????
E?B
?
0/1/0?????????
?
0/1/1?????????
? ?
G__inference_lstm_cell_1_layer_call_and_return_conditional_losses_164522?,.-??}
v?s
 ?
inputs????????? 
K?H
"?
states/0?????????
"?
states/1?????????
p
? "s?p
i?f
?
0/0?????????
E?B
?
0/1/0?????????
?
0/1/1?????????
? ?
,__inference_lstm_cell_1_layer_call_fn_164277?,.-??}
v?s
 ?
inputs????????? 
K?H
"?
states/0?????????
"?
states/1?????????
p 
? "c?`
?
0?????????
A?>
?
1/0?????????
?
1/1??????????
,__inference_lstm_cell_1_layer_call_fn_164294?,.-??}
v?s
 ?
inputs????????? 
K?H
"?
states/0?????????
"?
states/1?????????
p
? "c?`
?
0?????????
A?>
?
1/0?????????
?
1/1??????????
H__inference_sequential_9_layer_call_and_return_conditional_losses_162145~,.-!"K?H
A?>
4?1
embedding_9_input??????????????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_9_layer_call_and_return_conditional_losses_162170~,.-!"K?H
A?>
4?1
embedding_9_input??????????????????
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_9_layer_call_and_return_conditional_losses_162505s,.-!"@?=
6?3
)?&
inputs??????????????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_9_layer_call_and_return_conditional_losses_162904s,.-!"@?=
6?3
)?&
inputs??????????????????
p

 
? "%?"
?
0?????????
? ?
-__inference_sequential_9_layer_call_fn_161588q,.-!"K?H
A?>
4?1
embedding_9_input??????????????????
p 

 
? "???????????
-__inference_sequential_9_layer_call_fn_162120q,.-!"K?H
A?>
4?1
embedding_9_input??????????????????
p

 
? "???????????
-__inference_sequential_9_layer_call_fn_162220f,.-!"@?=
6?3
)?&
inputs??????????????????
p 

 
? "???????????
-__inference_sequential_9_layer_call_fn_162241f,.-!"@?=
6?3
)?&
inputs??????????????????
p

 
? "???????????
$__inference_signature_wrapper_162199?,.-!"X?U
? 
N?K
I
embedding_9_input4?1
embedding_9_input??????????????????"3?0
.
dense_19"?
dense_19?????????