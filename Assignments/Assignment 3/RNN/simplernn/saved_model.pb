??
??
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
?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28Ɔ
?
embedding_7/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?' *'
shared_nameembedding_7/embeddings
?
*embedding_7/embeddings/Read/ReadVariableOpReadVariableOpembedding_7/embeddings*
_output_shapes
:	?' *
dtype0
z
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_14/kernel
s
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes

:
*
dtype0
r
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_14/bias
k
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes
:
*
dtype0
z
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_15/kernel
s
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*
_output_shapes

:
*
dtype0
r
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_15/bias
k
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
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
%simple_rnn_7/simple_rnn_cell_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *6
shared_name'%simple_rnn_7/simple_rnn_cell_7/kernel
?
9simple_rnn_7/simple_rnn_cell_7/kernel/Read/ReadVariableOpReadVariableOp%simple_rnn_7/simple_rnn_cell_7/kernel*
_output_shapes

: *
dtype0
?
/simple_rnn_7/simple_rnn_cell_7/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*@
shared_name1/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel
?
Csimple_rnn_7/simple_rnn_cell_7/recurrent_kernel/Read/ReadVariableOpReadVariableOp/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel*
_output_shapes

:*
dtype0
?
#simple_rnn_7/simple_rnn_cell_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#simple_rnn_7/simple_rnn_cell_7/bias
?
7simple_rnn_7/simple_rnn_cell_7/bias/Read/ReadVariableOpReadVariableOp#simple_rnn_7/simple_rnn_cell_7/bias*
_output_shapes
:*
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
Adam/embedding_7/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?' *.
shared_nameAdam/embedding_7/embeddings/m
?
1Adam/embedding_7/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_7/embeddings/m*
_output_shapes
:	?' *
dtype0
?
Adam/dense_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/dense_14/kernel/m
?
*Adam/dense_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/m*
_output_shapes

:
*
dtype0
?
Adam/dense_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_14/bias/m
y
(Adam/dense_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/m*
_output_shapes
:
*
dtype0
?
Adam/dense_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/dense_15/kernel/m
?
*Adam/dense_15/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/m*
_output_shapes

:
*
dtype0
?
Adam/dense_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_15/bias/m
y
(Adam/dense_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_15/bias/m*
_output_shapes
:*
dtype0
?
,Adam/simple_rnn_7/simple_rnn_cell_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *=
shared_name.,Adam/simple_rnn_7/simple_rnn_cell_7/kernel/m
?
@Adam/simple_rnn_7/simple_rnn_cell_7/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_7/simple_rnn_cell_7/kernel/m*
_output_shapes

: *
dtype0
?
6Adam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*G
shared_name86Adam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/m
?
JAdam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/m*
_output_shapes

:*
dtype0
?
*Adam/simple_rnn_7/simple_rnn_cell_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/simple_rnn_7/simple_rnn_cell_7/bias/m
?
>Adam/simple_rnn_7/simple_rnn_cell_7/bias/m/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_7/simple_rnn_cell_7/bias/m*
_output_shapes
:*
dtype0
?
Adam/embedding_7/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?' *.
shared_nameAdam/embedding_7/embeddings/v
?
1Adam/embedding_7/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_7/embeddings/v*
_output_shapes
:	?' *
dtype0
?
Adam/dense_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/dense_14/kernel/v
?
*Adam/dense_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/v*
_output_shapes

:
*
dtype0
?
Adam/dense_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_14/bias/v
y
(Adam/dense_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/v*
_output_shapes
:
*
dtype0
?
Adam/dense_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/dense_15/kernel/v
?
*Adam/dense_15/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/v*
_output_shapes

:
*
dtype0
?
Adam/dense_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_15/bias/v
y
(Adam/dense_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_15/bias/v*
_output_shapes
:*
dtype0
?
,Adam/simple_rnn_7/simple_rnn_cell_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *=
shared_name.,Adam/simple_rnn_7/simple_rnn_cell_7/kernel/v
?
@Adam/simple_rnn_7/simple_rnn_cell_7/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_7/simple_rnn_cell_7/kernel/v*
_output_shapes

: *
dtype0
?
6Adam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*G
shared_name86Adam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/v
?
JAdam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/v*
_output_shapes

:*
dtype0
?
*Adam/simple_rnn_7/simple_rnn_cell_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/simple_rnn_7/simple_rnn_cell_7/bias/v
?
>Adam/simple_rnn_7/simple_rnn_cell_7/bias/v/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_7/simple_rnn_cell_7/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?4
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
+learning_ratembmcmd!me"mf,mg-mh.mivjvkvl!vm"vn,vo-vp.vq
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
VARIABLE_VALUEembedding_7/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE
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
~

,kernel
-recurrent_kernel
.bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
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

=states
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEdense_14/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_14/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
?
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEdense_15/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_15/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1

!0
"1
 
?
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
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
a_
VARIABLE_VALUE%simple_rnn_7/simple_rnn_cell_7/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#simple_rnn_7/simple_rnn_cell_7/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
 
#
0
1
2
3
4

R0
S1
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
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
9	variables
:trainable_variables
;regularization_losses
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
	Ytotal
	Zcount
[	variables
\	keras_api
D
	]total
	^count
_
_fn_kwargs
`	variables
a	keras_api
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
Y0
Z1

[	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

]0
^1

`	variables
??
VARIABLE_VALUEAdam/embedding_7/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_14/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_14/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_15/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_15/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/simple_rnn_7/simple_rnn_cell_7/kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/simple_rnn_7/simple_rnn_cell_7/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/embedding_7/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_14/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_14/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_15/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_15/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/simple_rnn_7/simple_rnn_cell_7/kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/simple_rnn_7/simple_rnn_cell_7/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
!serving_default_embedding_7_inputPlaceholder*0
_output_shapes
:??????????????????*
dtype0*%
shape:??????????????????
?
StatefulPartitionedCallStatefulPartitionedCall!serving_default_embedding_7_inputembedding_7/embeddings%simple_rnn_7/simple_rnn_cell_7/kernel#simple_rnn_7/simple_rnn_cell_7/bias/simple_rnn_7/simple_rnn_cell_7/recurrent_kerneldense_14/kerneldense_14/biasdense_15/kerneldense_15/bias*
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
$__inference_signature_wrapper_109627
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*embedding_7/embeddings/Read/ReadVariableOp#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOp#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp9simple_rnn_7/simple_rnn_cell_7/kernel/Read/ReadVariableOpCsimple_rnn_7/simple_rnn_cell_7/recurrent_kernel/Read/ReadVariableOp7simple_rnn_7/simple_rnn_cell_7/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp1Adam/embedding_7/embeddings/m/Read/ReadVariableOp*Adam/dense_14/kernel/m/Read/ReadVariableOp(Adam/dense_14/bias/m/Read/ReadVariableOp*Adam/dense_15/kernel/m/Read/ReadVariableOp(Adam/dense_15/bias/m/Read/ReadVariableOp@Adam/simple_rnn_7/simple_rnn_cell_7/kernel/m/Read/ReadVariableOpJAdam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/m/Read/ReadVariableOp>Adam/simple_rnn_7/simple_rnn_cell_7/bias/m/Read/ReadVariableOp1Adam/embedding_7/embeddings/v/Read/ReadVariableOp*Adam/dense_14/kernel/v/Read/ReadVariableOp(Adam/dense_14/bias/v/Read/ReadVariableOp*Adam/dense_15/kernel/v/Read/ReadVariableOp(Adam/dense_15/bias/v/Read/ReadVariableOp@Adam/simple_rnn_7/simple_rnn_cell_7/kernel/v/Read/ReadVariableOpJAdam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/v/Read/ReadVariableOp>Adam/simple_rnn_7/simple_rnn_cell_7/bias/v/Read/ReadVariableOpConst*.
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
__inference__traced_save_110902
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_7/embeddingsdense_14/kerneldense_14/biasdense_15/kerneldense_15/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate%simple_rnn_7/simple_rnn_cell_7/kernel/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel#simple_rnn_7/simple_rnn_cell_7/biastotalcounttotal_1count_1Adam/embedding_7/embeddings/mAdam/dense_14/kernel/mAdam/dense_14/bias/mAdam/dense_15/kernel/mAdam/dense_15/bias/m,Adam/simple_rnn_7/simple_rnn_cell_7/kernel/m6Adam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/m*Adam/simple_rnn_7/simple_rnn_cell_7/bias/mAdam/embedding_7/embeddings/vAdam/dense_14/kernel/vAdam/dense_14/bias/vAdam/dense_15/kernel/vAdam/dense_15/bias/v,Adam/simple_rnn_7/simple_rnn_cell_7/kernel/v6Adam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/v*Adam/simple_rnn_7/simple_rnn_cell_7/bias/v*-
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
"__inference__traced_restore_111011??
?
F
*__inference_dropout_7_layer_call_fn_110644

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
E__inference_dropout_7_layer_call_and_return_conditional_losses_109192`
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
??
?
H__inference_sequential_7_layer_call_and_return_conditional_losses_109998

inputs6
#embedding_7_embedding_lookup_109818:	?' O
=simple_rnn_7_simple_rnn_cell_7_matmul_readvariableop_resource: L
>simple_rnn_7_simple_rnn_cell_7_biasadd_readvariableop_resource:Q
?simple_rnn_7_simple_rnn_cell_7_matmul_1_readvariableop_resource:9
'dense_14_matmul_readvariableop_resource:
6
(dense_14_biasadd_readvariableop_resource:
9
'dense_15_matmul_readvariableop_resource:
6
(dense_15_biasadd_readvariableop_resource:
identity??dense_14/BiasAdd/ReadVariableOp?dense_14/MatMul/ReadVariableOp?dense_15/BiasAdd/ReadVariableOp?dense_15/MatMul/ReadVariableOp?embedding_7/embedding_lookup?5simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOp?4simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOp?6simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOp?simple_rnn_7/whilej
embedding_7/CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:???????????????????
embedding_7/embedding_lookupResourceGather#embedding_7_embedding_lookup_109818embedding_7/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding_7/embedding_lookup/109818*4
_output_shapes"
 :?????????????????? *
dtype0?
%embedding_7/embedding_lookup/IdentityIdentity%embedding_7/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_7/embedding_lookup/109818*4
_output_shapes"
 :?????????????????? ?
'embedding_7/embedding_lookup/Identity_1Identity.embedding_7/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :?????????????????? r
simple_rnn_7/ShapeShape0embedding_7/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:j
 simple_rnn_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_7/strided_sliceStridedSlicesimple_rnn_7/Shape:output:0)simple_rnn_7/strided_slice/stack:output:0+simple_rnn_7/strided_slice/stack_1:output:0+simple_rnn_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_7/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_7/zeros/packedPack#simple_rnn_7/strided_slice:output:0$simple_rnn_7/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_7/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
simple_rnn_7/zerosFill"simple_rnn_7/zeros/packed:output:0!simple_rnn_7/zeros/Const:output:0*
T0*'
_output_shapes
:?????????p
simple_rnn_7/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_7/transpose	Transpose0embedding_7/embedding_lookup/Identity_1:output:0$simple_rnn_7/transpose/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? ^
simple_rnn_7/Shape_1Shapesimple_rnn_7/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_7/strided_slice_1StridedSlicesimple_rnn_7/Shape_1:output:0+simple_rnn_7/strided_slice_1/stack:output:0-simple_rnn_7/strided_slice_1/stack_1:output:0-simple_rnn_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_7/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
simple_rnn_7/TensorArrayV2TensorListReserve1simple_rnn_7/TensorArrayV2/element_shape:output:0%simple_rnn_7/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Bsimple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
4simple_rnn_7/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_7/transpose:y:0Ksimple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???l
"simple_rnn_7/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_7/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_7/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_7/strided_slice_2StridedSlicesimple_rnn_7/transpose:y:0+simple_rnn_7/strided_slice_2/stack:output:0-simple_rnn_7/strided_slice_2/stack_1:output:0-simple_rnn_7/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask?
.simple_rnn_7/simple_rnn_cell_7/ones_like/ShapeShape%simple_rnn_7/strided_slice_2:output:0*
T0*
_output_shapes
:s
.simple_rnn_7/simple_rnn_cell_7/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
(simple_rnn_7/simple_rnn_cell_7/ones_likeFill7simple_rnn_7/simple_rnn_cell_7/ones_like/Shape:output:07simple_rnn_7/simple_rnn_cell_7/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? q
,simple_rnn_7/simple_rnn_cell_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
*simple_rnn_7/simple_rnn_cell_7/dropout/MulMul1simple_rnn_7/simple_rnn_cell_7/ones_like:output:05simple_rnn_7/simple_rnn_cell_7/dropout/Const:output:0*
T0*'
_output_shapes
:????????? ?
,simple_rnn_7/simple_rnn_cell_7/dropout/ShapeShape1simple_rnn_7/simple_rnn_cell_7/ones_like:output:0*
T0*
_output_shapes
:?
Csimple_rnn_7/simple_rnn_cell_7/dropout/random_uniform/RandomUniformRandomUniform5simple_rnn_7/simple_rnn_cell_7/dropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2??Qz
5simple_rnn_7/simple_rnn_cell_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
3simple_rnn_7/simple_rnn_cell_7/dropout/GreaterEqualGreaterEqualLsimple_rnn_7/simple_rnn_cell_7/dropout/random_uniform/RandomUniform:output:0>simple_rnn_7/simple_rnn_cell_7/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
+simple_rnn_7/simple_rnn_cell_7/dropout/CastCast7simple_rnn_7/simple_rnn_cell_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
,simple_rnn_7/simple_rnn_cell_7/dropout/Mul_1Mul.simple_rnn_7/simple_rnn_cell_7/dropout/Mul:z:0/simple_rnn_7/simple_rnn_cell_7/dropout/Cast:y:0*
T0*'
_output_shapes
:????????? {
0simple_rnn_7/simple_rnn_cell_7/ones_like_1/ShapeShapesimple_rnn_7/zeros:output:0*
T0*
_output_shapes
:u
0simple_rnn_7/simple_rnn_cell_7/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
*simple_rnn_7/simple_rnn_cell_7/ones_like_1Fill9simple_rnn_7/simple_rnn_cell_7/ones_like_1/Shape:output:09simple_rnn_7/simple_rnn_cell_7/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????s
.simple_rnn_7/simple_rnn_cell_7/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
,simple_rnn_7/simple_rnn_cell_7/dropout_1/MulMul3simple_rnn_7/simple_rnn_cell_7/ones_like_1:output:07simple_rnn_7/simple_rnn_cell_7/dropout_1/Const:output:0*
T0*'
_output_shapes
:??????????
.simple_rnn_7/simple_rnn_cell_7/dropout_1/ShapeShape3simple_rnn_7/simple_rnn_cell_7/ones_like_1:output:0*
T0*
_output_shapes
:?
Esimple_rnn_7/simple_rnn_cell_7/dropout_1/random_uniform/RandomUniformRandomUniform7simple_rnn_7/simple_rnn_cell_7/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???|
7simple_rnn_7/simple_rnn_cell_7/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
5simple_rnn_7/simple_rnn_cell_7/dropout_1/GreaterEqualGreaterEqualNsimple_rnn_7/simple_rnn_cell_7/dropout_1/random_uniform/RandomUniform:output:0@simple_rnn_7/simple_rnn_cell_7/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
-simple_rnn_7/simple_rnn_cell_7/dropout_1/CastCast9simple_rnn_7/simple_rnn_cell_7/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
.simple_rnn_7/simple_rnn_cell_7/dropout_1/Mul_1Mul0simple_rnn_7/simple_rnn_cell_7/dropout_1/Mul:z:01simple_rnn_7/simple_rnn_cell_7/dropout_1/Cast:y:0*
T0*'
_output_shapes
:??????????
"simple_rnn_7/simple_rnn_cell_7/mulMul%simple_rnn_7/strided_slice_2:output:00simple_rnn_7/simple_rnn_cell_7/dropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
4simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOp=simple_rnn_7_simple_rnn_cell_7_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
%simple_rnn_7/simple_rnn_cell_7/MatMulMatMul&simple_rnn_7/simple_rnn_cell_7/mul:z:0<simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
5simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_7_simple_rnn_cell_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
&simple_rnn_7/simple_rnn_cell_7/BiasAddBiasAdd/simple_rnn_7/simple_rnn_cell_7/MatMul:product:0=simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
$simple_rnn_7/simple_rnn_cell_7/mul_1Mulsimple_rnn_7/zeros:output:02simple_rnn_7/simple_rnn_cell_7/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:??????????
6simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_7_simple_rnn_cell_7_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0?
'simple_rnn_7/simple_rnn_cell_7/MatMul_1MatMul(simple_rnn_7/simple_rnn_cell_7/mul_1:z:0>simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
"simple_rnn_7/simple_rnn_cell_7/addAddV2/simple_rnn_7/simple_rnn_cell_7/BiasAdd:output:01simple_rnn_7/simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:??????????
#simple_rnn_7/simple_rnn_cell_7/TanhTanh&simple_rnn_7/simple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:?????????{
*simple_rnn_7/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
simple_rnn_7/TensorArrayV2_1TensorListReserve3simple_rnn_7/TensorArrayV2_1/element_shape:output:0%simple_rnn_7/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???S
simple_rnn_7/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_7/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????a
simple_rnn_7/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
simple_rnn_7/whileWhile(simple_rnn_7/while/loop_counter:output:0.simple_rnn_7/while/maximum_iterations:output:0simple_rnn_7/time:output:0%simple_rnn_7/TensorArrayV2_1:handle:0simple_rnn_7/zeros:output:0%simple_rnn_7/strided_slice_1:output:0Dsimple_rnn_7/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_7_simple_rnn_cell_7_matmul_readvariableop_resource>simple_rnn_7_simple_rnn_cell_7_biasadd_readvariableop_resource?simple_rnn_7_simple_rnn_cell_7_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( **
body"R 
simple_rnn_7_while_body_109886**
cond"R 
simple_rnn_7_while_cond_109885*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations ?
=simple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
/simple_rnn_7/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_7/while:output:3Fsimple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0u
"simple_rnn_7/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????n
$simple_rnn_7/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_7/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_7/strided_slice_3StridedSlice8simple_rnn_7/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_7/strided_slice_3/stack:output:0-simple_rnn_7/strided_slice_3/stack_1:output:0-simple_rnn_7/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskr
simple_rnn_7/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_7/transpose_1	Transpose8simple_rnn_7/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_7/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :???????????????????
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
dense_14/MatMulMatMul%simple_rnn_7/strided_slice_3:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
b
dense_14/ReluReludense_14/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
\
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_7/dropout/MulMuldense_14/Relu:activations:0 dropout_7/dropout/Const:output:0*
T0*'
_output_shapes
:?????????
b
dropout_7/dropout/ShapeShapedense_14/Relu:activations:0*
T0*
_output_shapes
:?
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????
*
dtype0e
 dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_7/dropout/GreaterEqualGreaterEqual7dropout_7/dropout/random_uniform/RandomUniform:output:0)dropout_7/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????
?
dropout_7/dropout/CastCast"dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
?
dropout_7/dropout/Mul_1Muldropout_7/dropout/Mul:z:0dropout_7/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????
?
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
dense_15/MatMulMatMuldropout_7/dropout/Mul_1:z:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_15/SigmoidSigmoiddense_15/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
IdentityIdentitydense_15/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp^embedding_7/embedding_lookup6^simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOp5^simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOp7^simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOp^simple_rnn_7/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????????????: : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2<
embedding_7/embedding_lookupembedding_7/embedding_lookup2n
5simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOp5simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOp2l
4simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOp4simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOp2p
6simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOp6simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOp2(
simple_rnn_7/whilesimple_rnn_7/while:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?	
?
$__inference_signature_wrapper_109627
embedding_7_input
unknown:	?' 
	unknown_0: 
	unknown_1:
	unknown_2:
	unknown_3:

	unknown_4:

	unknown_5:

	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallembedding_7_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
!__inference__wrapped_model_108694o
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
_user_specified_nameembedding_7_input
?
?
while_cond_108945
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_108945___redundant_placeholder04
0while_while_cond_108945___redundant_placeholder14
0while_while_cond_108945___redundant_placeholder24
0while_while_cond_108945___redundant_placeholder3
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
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????: ::::: 
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
:?????????:

_output_shapes
: :

_output_shapes
:
?F
?
H__inference_simple_rnn_7_layer_call_and_return_conditional_losses_110463

inputsB
0simple_rnn_cell_7_matmul_readvariableop_resource: ?
1simple_rnn_cell_7_biasadd_readvariableop_resource:D
2simple_rnn_cell_7_matmul_1_readvariableop_resource:
identity??(simple_rnn_cell_7/BiasAdd/ReadVariableOp?'simple_rnn_cell_7/MatMul/ReadVariableOp?)simple_rnn_cell_7/MatMul_1/ReadVariableOp?while;
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
shrink_axis_maski
!simple_rnn_cell_7/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:f
!simple_rnn_cell_7/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
simple_rnn_cell_7/ones_likeFill*simple_rnn_cell_7/ones_like/Shape:output:0*simple_rnn_cell_7/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? a
#simple_rnn_cell_7/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:h
#simple_rnn_cell_7/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
simple_rnn_cell_7/ones_like_1Fill,simple_rnn_cell_7/ones_like_1/Shape:output:0,simple_rnn_cell_7/ones_like_1/Const:output:0*
T0*'
_output_shapes
:??????????
simple_rnn_cell_7/mulMulstrided_slice_2:output:0$simple_rnn_cell_7/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
'simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_7_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
simple_rnn_cell_7/MatMulMatMulsimple_rnn_cell_7/mul:z:0/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
(simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
simple_rnn_cell_7/BiasAddBiasAdd"simple_rnn_cell_7/MatMul:product:00simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
simple_rnn_cell_7/mul_1Mulzeros:output:0&simple_rnn_cell_7/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
)simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_7_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0?
simple_rnn_cell_7/MatMul_1MatMulsimple_rnn_cell_7/mul_1:z:01simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
simple_rnn_cell_7/addAddV2"simple_rnn_cell_7/BiasAdd:output:0$simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:?????????k
simple_rnn_cell_7/TanhTanhsimple_rnn_cell_7/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_7_matmul_readvariableop_resource1simple_rnn_cell_7_biasadd_readvariableop_resource2simple_rnn_cell_7_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_110389*
condR
while_cond_110388*8
output_shapes'
%: : : : :?????????: : : : : *
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
 :??????????????????g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp)^simple_rnn_cell_7/BiasAdd/ReadVariableOp(^simple_rnn_cell_7/MatMul/ReadVariableOp*^simple_rnn_cell_7/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????????????? : : : 2T
(simple_rnn_cell_7/BiasAdd/ReadVariableOp(simple_rnn_cell_7/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_7/MatMul/ReadVariableOp'simple_rnn_cell_7/MatMul/ReadVariableOp2V
)simple_rnn_cell_7/MatMul_1/ReadVariableOp)simple_rnn_cell_7/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?
?
H__inference_sequential_7_layer_call_and_return_conditional_losses_109508

inputs%
embedding_7_109486:	?' %
simple_rnn_7_109489: !
simple_rnn_7_109491:%
simple_rnn_7_109493:!
dense_14_109496:

dense_14_109498:
!
dense_15_109502:

dense_15_109504:
identity?? dense_14/StatefulPartitionedCall? dense_15/StatefulPartitionedCall?!dropout_7/StatefulPartitionedCall?#embedding_7/StatefulPartitionedCall?$simple_rnn_7/StatefulPartitionedCall?
#embedding_7/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_7_109486*
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
G__inference_embedding_7_layer_call_and_return_conditional_losses_109035?
$simple_rnn_7/StatefulPartitionedCallStatefulPartitionedCall,embedding_7/StatefulPartitionedCall:output:0simple_rnn_7_109489simple_rnn_7_109491simple_rnn_7_109493*
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
GPU2*0J 8? *Q
fLRJ
H__inference_simple_rnn_7_layer_call_and_return_conditional_losses_109443?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_7/StatefulPartitionedCall:output:0dense_14_109496dense_14_109498*
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
D__inference_dense_14_layer_call_and_return_conditional_losses_109181?
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
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
E__inference_dropout_7_layer_call_and_return_conditional_losses_109261?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0dense_15_109502dense_15_109504*
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
D__inference_dense_15_layer_call_and_return_conditional_losses_109205x
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall$^embedding_7/StatefulPartitionedCall%^simple_rnn_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????????????: : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2J
#embedding_7/StatefulPartitionedCall#embedding_7/StatefulPartitionedCall2L
$simple_rnn_7/StatefulPartitionedCall$simple_rnn_7/StatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
??
?	
!__inference__wrapped_model_108694
embedding_7_inputC
0sequential_7_embedding_7_embedding_lookup_108553:	?' \
Jsequential_7_simple_rnn_7_simple_rnn_cell_7_matmul_readvariableop_resource: Y
Ksequential_7_simple_rnn_7_simple_rnn_cell_7_biasadd_readvariableop_resource:^
Lsequential_7_simple_rnn_7_simple_rnn_cell_7_matmul_1_readvariableop_resource:F
4sequential_7_dense_14_matmul_readvariableop_resource:
C
5sequential_7_dense_14_biasadd_readvariableop_resource:
F
4sequential_7_dense_15_matmul_readvariableop_resource:
C
5sequential_7_dense_15_biasadd_readvariableop_resource:
identity??,sequential_7/dense_14/BiasAdd/ReadVariableOp?+sequential_7/dense_14/MatMul/ReadVariableOp?,sequential_7/dense_15/BiasAdd/ReadVariableOp?+sequential_7/dense_15/MatMul/ReadVariableOp?)sequential_7/embedding_7/embedding_lookup?Bsequential_7/simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOp?Asequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOp?Csequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOp?sequential_7/simple_rnn_7/while?
sequential_7/embedding_7/CastCastembedding_7_input*

DstT0*

SrcT0*0
_output_shapes
:???????????????????
)sequential_7/embedding_7/embedding_lookupResourceGather0sequential_7_embedding_7_embedding_lookup_108553!sequential_7/embedding_7/Cast:y:0*
Tindices0*C
_class9
75loc:@sequential_7/embedding_7/embedding_lookup/108553*4
_output_shapes"
 :?????????????????? *
dtype0?
2sequential_7/embedding_7/embedding_lookup/IdentityIdentity2sequential_7/embedding_7/embedding_lookup:output:0*
T0*C
_class9
75loc:@sequential_7/embedding_7/embedding_lookup/108553*4
_output_shapes"
 :?????????????????? ?
4sequential_7/embedding_7/embedding_lookup/Identity_1Identity;sequential_7/embedding_7/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :?????????????????? ?
sequential_7/simple_rnn_7/ShapeShape=sequential_7/embedding_7/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:w
-sequential_7/simple_rnn_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/sequential_7/simple_rnn_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/sequential_7/simple_rnn_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'sequential_7/simple_rnn_7/strided_sliceStridedSlice(sequential_7/simple_rnn_7/Shape:output:06sequential_7/simple_rnn_7/strided_slice/stack:output:08sequential_7/simple_rnn_7/strided_slice/stack_1:output:08sequential_7/simple_rnn_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_7/simple_rnn_7/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :?
&sequential_7/simple_rnn_7/zeros/packedPack0sequential_7/simple_rnn_7/strided_slice:output:01sequential_7/simple_rnn_7/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:j
%sequential_7/simple_rnn_7/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential_7/simple_rnn_7/zerosFill/sequential_7/simple_rnn_7/zeros/packed:output:0.sequential_7/simple_rnn_7/zeros/Const:output:0*
T0*'
_output_shapes
:?????????}
(sequential_7/simple_rnn_7/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
#sequential_7/simple_rnn_7/transpose	Transpose=sequential_7/embedding_7/embedding_lookup/Identity_1:output:01sequential_7/simple_rnn_7/transpose/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? x
!sequential_7/simple_rnn_7/Shape_1Shape'sequential_7/simple_rnn_7/transpose:y:0*
T0*
_output_shapes
:y
/sequential_7/simple_rnn_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_7/simple_rnn_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_7/simple_rnn_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential_7/simple_rnn_7/strided_slice_1StridedSlice*sequential_7/simple_rnn_7/Shape_1:output:08sequential_7/simple_rnn_7/strided_slice_1/stack:output:0:sequential_7/simple_rnn_7/strided_slice_1/stack_1:output:0:sequential_7/simple_rnn_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
5sequential_7/simple_rnn_7/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
'sequential_7/simple_rnn_7/TensorArrayV2TensorListReserve>sequential_7/simple_rnn_7/TensorArrayV2/element_shape:output:02sequential_7/simple_rnn_7/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Osequential_7/simple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
Asequential_7/simple_rnn_7/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor'sequential_7/simple_rnn_7/transpose:y:0Xsequential_7/simple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???y
/sequential_7/simple_rnn_7/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_7/simple_rnn_7/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_7/simple_rnn_7/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential_7/simple_rnn_7/strided_slice_2StridedSlice'sequential_7/simple_rnn_7/transpose:y:08sequential_7/simple_rnn_7/strided_slice_2/stack:output:0:sequential_7/simple_rnn_7/strided_slice_2/stack_1:output:0:sequential_7/simple_rnn_7/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask?
;sequential_7/simple_rnn_7/simple_rnn_cell_7/ones_like/ShapeShape2sequential_7/simple_rnn_7/strided_slice_2:output:0*
T0*
_output_shapes
:?
;sequential_7/simple_rnn_7/simple_rnn_cell_7/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
5sequential_7/simple_rnn_7/simple_rnn_cell_7/ones_likeFillDsequential_7/simple_rnn_7/simple_rnn_cell_7/ones_like/Shape:output:0Dsequential_7/simple_rnn_7/simple_rnn_cell_7/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? ?
=sequential_7/simple_rnn_7/simple_rnn_cell_7/ones_like_1/ShapeShape(sequential_7/simple_rnn_7/zeros:output:0*
T0*
_output_shapes
:?
=sequential_7/simple_rnn_7/simple_rnn_cell_7/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
7sequential_7/simple_rnn_7/simple_rnn_cell_7/ones_like_1FillFsequential_7/simple_rnn_7/simple_rnn_cell_7/ones_like_1/Shape:output:0Fsequential_7/simple_rnn_7/simple_rnn_cell_7/ones_like_1/Const:output:0*
T0*'
_output_shapes
:??????????
/sequential_7/simple_rnn_7/simple_rnn_cell_7/mulMul2sequential_7/simple_rnn_7/strided_slice_2:output:0>sequential_7/simple_rnn_7/simple_rnn_cell_7/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
Asequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOpJsequential_7_simple_rnn_7_simple_rnn_cell_7_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
2sequential_7/simple_rnn_7/simple_rnn_cell_7/MatMulMatMul3sequential_7/simple_rnn_7/simple_rnn_cell_7/mul:z:0Isequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Bsequential_7/simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOpKsequential_7_simple_rnn_7_simple_rnn_cell_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
3sequential_7/simple_rnn_7/simple_rnn_cell_7/BiasAddBiasAdd<sequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul:product:0Jsequential_7/simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
1sequential_7/simple_rnn_7/simple_rnn_cell_7/mul_1Mul(sequential_7/simple_rnn_7/zeros:output:0@sequential_7/simple_rnn_7/simple_rnn_cell_7/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
Csequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOpLsequential_7_simple_rnn_7_simple_rnn_cell_7_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0?
4sequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul_1MatMul5sequential_7/simple_rnn_7/simple_rnn_cell_7/mul_1:z:0Ksequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
/sequential_7/simple_rnn_7/simple_rnn_cell_7/addAddV2<sequential_7/simple_rnn_7/simple_rnn_cell_7/BiasAdd:output:0>sequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:??????????
0sequential_7/simple_rnn_7/simple_rnn_cell_7/TanhTanh3sequential_7/simple_rnn_7/simple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:??????????
7sequential_7/simple_rnn_7/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)sequential_7/simple_rnn_7/TensorArrayV2_1TensorListReserve@sequential_7/simple_rnn_7/TensorArrayV2_1/element_shape:output:02sequential_7/simple_rnn_7/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???`
sequential_7/simple_rnn_7/timeConst*
_output_shapes
: *
dtype0*
value	B : }
2sequential_7/simple_rnn_7/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????n
,sequential_7/simple_rnn_7/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
sequential_7/simple_rnn_7/whileWhile5sequential_7/simple_rnn_7/while/loop_counter:output:0;sequential_7/simple_rnn_7/while/maximum_iterations:output:0'sequential_7/simple_rnn_7/time:output:02sequential_7/simple_rnn_7/TensorArrayV2_1:handle:0(sequential_7/simple_rnn_7/zeros:output:02sequential_7/simple_rnn_7/strided_slice_1:output:0Qsequential_7/simple_rnn_7/TensorArrayUnstack/TensorListFromTensor:output_handle:0Jsequential_7_simple_rnn_7_simple_rnn_cell_7_matmul_readvariableop_resourceKsequential_7_simple_rnn_7_simple_rnn_cell_7_biasadd_readvariableop_resourceLsequential_7_simple_rnn_7_simple_rnn_cell_7_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *7
body/R-
+sequential_7_simple_rnn_7_while_body_108605*7
cond/R-
+sequential_7_simple_rnn_7_while_cond_108604*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations ?
Jsequential_7/simple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
<sequential_7/simple_rnn_7/TensorArrayV2Stack/TensorListStackTensorListStack(sequential_7/simple_rnn_7/while:output:3Ssequential_7/simple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0?
/sequential_7/simple_rnn_7/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????{
1sequential_7/simple_rnn_7/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1sequential_7/simple_rnn_7/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential_7/simple_rnn_7/strided_slice_3StridedSliceEsequential_7/simple_rnn_7/TensorArrayV2Stack/TensorListStack:tensor:08sequential_7/simple_rnn_7/strided_slice_3/stack:output:0:sequential_7/simple_rnn_7/strided_slice_3/stack_1:output:0:sequential_7/simple_rnn_7/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask
*sequential_7/simple_rnn_7/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
%sequential_7/simple_rnn_7/transpose_1	TransposeEsequential_7/simple_rnn_7/TensorArrayV2Stack/TensorListStack:tensor:03sequential_7/simple_rnn_7/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :???????????????????
+sequential_7/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_14_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
sequential_7/dense_14/MatMulMatMul2sequential_7/simple_rnn_7/strided_slice_3:output:03sequential_7/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
,sequential_7/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_14_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
sequential_7/dense_14/BiasAddBiasAdd&sequential_7/dense_14/MatMul:product:04sequential_7/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
|
sequential_7/dense_14/ReluRelu&sequential_7/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
sequential_7/dropout_7/IdentityIdentity(sequential_7/dense_14/Relu:activations:0*
T0*'
_output_shapes
:?????????
?
+sequential_7/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_15_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
sequential_7/dense_15/MatMulMatMul(sequential_7/dropout_7/Identity:output:03sequential_7/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
,sequential_7/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_7/dense_15/BiasAddBiasAdd&sequential_7/dense_15/MatMul:product:04sequential_7/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
sequential_7/dense_15/SigmoidSigmoid&sequential_7/dense_15/BiasAdd:output:0*
T0*'
_output_shapes
:?????????p
IdentityIdentity!sequential_7/dense_15/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp-^sequential_7/dense_14/BiasAdd/ReadVariableOp,^sequential_7/dense_14/MatMul/ReadVariableOp-^sequential_7/dense_15/BiasAdd/ReadVariableOp,^sequential_7/dense_15/MatMul/ReadVariableOp*^sequential_7/embedding_7/embedding_lookupC^sequential_7/simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOpB^sequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOpD^sequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOp ^sequential_7/simple_rnn_7/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????????????: : : : : : : : 2\
,sequential_7/dense_14/BiasAdd/ReadVariableOp,sequential_7/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_14/MatMul/ReadVariableOp+sequential_7/dense_14/MatMul/ReadVariableOp2\
,sequential_7/dense_15/BiasAdd/ReadVariableOp,sequential_7/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_15/MatMul/ReadVariableOp+sequential_7/dense_15/MatMul/ReadVariableOp2V
)sequential_7/embedding_7/embedding_lookup)sequential_7/embedding_7/embedding_lookup2?
Bsequential_7/simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOpBsequential_7/simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOp2?
Asequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOpAsequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOp2?
Csequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOpCsequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOp2B
sequential_7/simple_rnn_7/whilesequential_7/simple_rnn_7/while:c _
0
_output_shapes
:??????????????????
+
_user_specified_nameembedding_7_input
?I
?
__inference__traced_save_110902
file_prefix5
1savev2_embedding_7_embeddings_read_readvariableop.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopD
@savev2_simple_rnn_7_simple_rnn_cell_7_kernel_read_readvariableopN
Jsavev2_simple_rnn_7_simple_rnn_cell_7_recurrent_kernel_read_readvariableopB
>savev2_simple_rnn_7_simple_rnn_cell_7_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop<
8savev2_adam_embedding_7_embeddings_m_read_readvariableop5
1savev2_adam_dense_14_kernel_m_read_readvariableop3
/savev2_adam_dense_14_bias_m_read_readvariableop5
1savev2_adam_dense_15_kernel_m_read_readvariableop3
/savev2_adam_dense_15_bias_m_read_readvariableopK
Gsavev2_adam_simple_rnn_7_simple_rnn_cell_7_kernel_m_read_readvariableopU
Qsavev2_adam_simple_rnn_7_simple_rnn_cell_7_recurrent_kernel_m_read_readvariableopI
Esavev2_adam_simple_rnn_7_simple_rnn_cell_7_bias_m_read_readvariableop<
8savev2_adam_embedding_7_embeddings_v_read_readvariableop5
1savev2_adam_dense_14_kernel_v_read_readvariableop3
/savev2_adam_dense_14_bias_v_read_readvariableop5
1savev2_adam_dense_15_kernel_v_read_readvariableop3
/savev2_adam_dense_15_bias_v_read_readvariableopK
Gsavev2_adam_simple_rnn_7_simple_rnn_cell_7_kernel_v_read_readvariableopU
Qsavev2_adam_simple_rnn_7_simple_rnn_cell_7_recurrent_kernel_v_read_readvariableopI
Esavev2_adam_simple_rnn_7_simple_rnn_cell_7_bias_v_read_readvariableop
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
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_embedding_7_embeddings_read_readvariableop*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop@savev2_simple_rnn_7_simple_rnn_cell_7_kernel_read_readvariableopJsavev2_simple_rnn_7_simple_rnn_cell_7_recurrent_kernel_read_readvariableop>savev2_simple_rnn_7_simple_rnn_cell_7_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop8savev2_adam_embedding_7_embeddings_m_read_readvariableop1savev2_adam_dense_14_kernel_m_read_readvariableop/savev2_adam_dense_14_bias_m_read_readvariableop1savev2_adam_dense_15_kernel_m_read_readvariableop/savev2_adam_dense_15_bias_m_read_readvariableopGsavev2_adam_simple_rnn_7_simple_rnn_cell_7_kernel_m_read_readvariableopQsavev2_adam_simple_rnn_7_simple_rnn_cell_7_recurrent_kernel_m_read_readvariableopEsavev2_adam_simple_rnn_7_simple_rnn_cell_7_bias_m_read_readvariableop8savev2_adam_embedding_7_embeddings_v_read_readvariableop1savev2_adam_dense_14_kernel_v_read_readvariableop/savev2_adam_dense_14_bias_v_read_readvariableop1savev2_adam_dense_15_kernel_v_read_readvariableop/savev2_adam_dense_15_bias_v_read_readvariableopGsavev2_adam_simple_rnn_7_simple_rnn_cell_7_kernel_v_read_readvariableopQsavev2_adam_simple_rnn_7_simple_rnn_cell_7_recurrent_kernel_v_read_readvariableopEsavev2_adam_simple_rnn_7_simple_rnn_cell_7_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
:: : : : : : ::: : : : :	?' :
:
:
:: :::	?' :
:
:
:: ::: 2(
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

: :$ 

_output_shapes

:: 

_output_shapes
::
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

: :$ 

_output_shapes

:: 

_output_shapes
::%!

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

: :$  

_output_shapes

:: !

_output_shapes
::"

_output_shapes
: 
?
c
E__inference_dropout_7_layer_call_and_return_conditional_losses_109192

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
??
?
"__inference__traced_restore_111011
file_prefix:
'assignvariableop_embedding_7_embeddings:	?' 4
"assignvariableop_1_dense_14_kernel:
.
 assignvariableop_2_dense_14_bias:
4
"assignvariableop_3_dense_15_kernel:
.
 assignvariableop_4_dense_15_bias:&
assignvariableop_5_adam_iter:	 (
assignvariableop_6_adam_beta_1: (
assignvariableop_7_adam_beta_2: '
assignvariableop_8_adam_decay: /
%assignvariableop_9_adam_learning_rate: K
9assignvariableop_10_simple_rnn_7_simple_rnn_cell_7_kernel: U
Cassignvariableop_11_simple_rnn_7_simple_rnn_cell_7_recurrent_kernel:E
7assignvariableop_12_simple_rnn_7_simple_rnn_cell_7_bias:#
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: D
1assignvariableop_17_adam_embedding_7_embeddings_m:	?' <
*assignvariableop_18_adam_dense_14_kernel_m:
6
(assignvariableop_19_adam_dense_14_bias_m:
<
*assignvariableop_20_adam_dense_15_kernel_m:
6
(assignvariableop_21_adam_dense_15_bias_m:R
@assignvariableop_22_adam_simple_rnn_7_simple_rnn_cell_7_kernel_m: \
Jassignvariableop_23_adam_simple_rnn_7_simple_rnn_cell_7_recurrent_kernel_m:L
>assignvariableop_24_adam_simple_rnn_7_simple_rnn_cell_7_bias_m:D
1assignvariableop_25_adam_embedding_7_embeddings_v:	?' <
*assignvariableop_26_adam_dense_14_kernel_v:
6
(assignvariableop_27_adam_dense_14_bias_v:
<
*assignvariableop_28_adam_dense_15_kernel_v:
6
(assignvariableop_29_adam_dense_15_bias_v:R
@assignvariableop_30_adam_simple_rnn_7_simple_rnn_cell_7_kernel_v: \
Jassignvariableop_31_adam_simple_rnn_7_simple_rnn_cell_7_recurrent_kernel_v:L
>assignvariableop_32_adam_simple_rnn_7_simple_rnn_cell_7_bias_v:
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
AssignVariableOpAssignVariableOp'assignvariableop_embedding_7_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_14_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp assignvariableop_2_dense_14_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_15_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense_15_biasIdentity_4:output:0"/device:CPU:0*
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
AssignVariableOp_10AssignVariableOp9assignvariableop_10_simple_rnn_7_simple_rnn_cell_7_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpCassignvariableop_11_simple_rnn_7_simple_rnn_cell_7_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp7assignvariableop_12_simple_rnn_7_simple_rnn_cell_7_biasIdentity_12:output:0"/device:CPU:0*
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
AssignVariableOp_17AssignVariableOp1assignvariableop_17_adam_embedding_7_embeddings_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_dense_14_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_dense_14_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_dense_15_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_dense_15_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp@assignvariableop_22_adam_simple_rnn_7_simple_rnn_cell_7_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpJassignvariableop_23_adam_simple_rnn_7_simple_rnn_cell_7_recurrent_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp>assignvariableop_24_adam_simple_rnn_7_simple_rnn_cell_7_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp1assignvariableop_25_adam_embedding_7_embeddings_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_dense_14_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_dense_14_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_dense_15_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_dense_15_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp@assignvariableop_30_adam_simple_rnn_7_simple_rnn_cell_7_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOpJassignvariableop_31_adam_simple_rnn_7_simple_rnn_cell_7_recurrent_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp>assignvariableop_32_adam_simple_rnn_7_simple_rnn_cell_7_bias_vIdentity_32:output:0"/device:CPU:0*
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
?Y
?
H__inference_simple_rnn_7_layer_call_and_return_conditional_losses_110619

inputsB
0simple_rnn_cell_7_matmul_readvariableop_resource: ?
1simple_rnn_cell_7_biasadd_readvariableop_resource:D
2simple_rnn_cell_7_matmul_1_readvariableop_resource:
identity??(simple_rnn_cell_7/BiasAdd/ReadVariableOp?'simple_rnn_cell_7/MatMul/ReadVariableOp?)simple_rnn_cell_7/MatMul_1/ReadVariableOp?while;
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
shrink_axis_maski
!simple_rnn_cell_7/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:f
!simple_rnn_cell_7/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
simple_rnn_cell_7/ones_likeFill*simple_rnn_cell_7/ones_like/Shape:output:0*simple_rnn_cell_7/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? d
simple_rnn_cell_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
simple_rnn_cell_7/dropout/MulMul$simple_rnn_cell_7/ones_like:output:0(simple_rnn_cell_7/dropout/Const:output:0*
T0*'
_output_shapes
:????????? s
simple_rnn_cell_7/dropout/ShapeShape$simple_rnn_cell_7/ones_like:output:0*
T0*
_output_shapes
:?
6simple_rnn_cell_7/dropout/random_uniform/RandomUniformRandomUniform(simple_rnn_cell_7/dropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2ݞ?m
(simple_rnn_cell_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
&simple_rnn_cell_7/dropout/GreaterEqualGreaterEqual?simple_rnn_cell_7/dropout/random_uniform/RandomUniform:output:01simple_rnn_cell_7/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
simple_rnn_cell_7/dropout/CastCast*simple_rnn_cell_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
simple_rnn_cell_7/dropout/Mul_1Mul!simple_rnn_cell_7/dropout/Mul:z:0"simple_rnn_cell_7/dropout/Cast:y:0*
T0*'
_output_shapes
:????????? a
#simple_rnn_cell_7/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:h
#simple_rnn_cell_7/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
simple_rnn_cell_7/ones_like_1Fill,simple_rnn_cell_7/ones_like_1/Shape:output:0,simple_rnn_cell_7/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????f
!simple_rnn_cell_7/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
simple_rnn_cell_7/dropout_1/MulMul&simple_rnn_cell_7/ones_like_1:output:0*simple_rnn_cell_7/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????w
!simple_rnn_cell_7/dropout_1/ShapeShape&simple_rnn_cell_7/ones_like_1:output:0*
T0*
_output_shapes
:?
8simple_rnn_cell_7/dropout_1/random_uniform/RandomUniformRandomUniform*simple_rnn_cell_7/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???o
*simple_rnn_cell_7/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(simple_rnn_cell_7/dropout_1/GreaterEqualGreaterEqualAsimple_rnn_cell_7/dropout_1/random_uniform/RandomUniform:output:03simple_rnn_cell_7/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
 simple_rnn_cell_7/dropout_1/CastCast,simple_rnn_cell_7/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
!simple_rnn_cell_7/dropout_1/Mul_1Mul#simple_rnn_cell_7/dropout_1/Mul:z:0$simple_rnn_cell_7/dropout_1/Cast:y:0*
T0*'
_output_shapes
:??????????
simple_rnn_cell_7/mulMulstrided_slice_2:output:0#simple_rnn_cell_7/dropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
'simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_7_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
simple_rnn_cell_7/MatMulMatMulsimple_rnn_cell_7/mul:z:0/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
(simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
simple_rnn_cell_7/BiasAddBiasAdd"simple_rnn_cell_7/MatMul:product:00simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
simple_rnn_cell_7/mul_1Mulzeros:output:0%simple_rnn_cell_7/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:??????????
)simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_7_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0?
simple_rnn_cell_7/MatMul_1MatMulsimple_rnn_cell_7/mul_1:z:01simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
simple_rnn_cell_7/addAddV2"simple_rnn_cell_7/BiasAdd:output:0$simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:?????????k
simple_rnn_cell_7/TanhTanhsimple_rnn_cell_7/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_7_matmul_readvariableop_resource1simple_rnn_cell_7_biasadd_readvariableop_resource2simple_rnn_cell_7_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_110529*
condR
while_cond_110528*8
output_shapes'
%: : : : :?????????: : : : : *
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
 :??????????????????g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp)^simple_rnn_cell_7/BiasAdd/ReadVariableOp(^simple_rnn_cell_7/MatMul/ReadVariableOp*^simple_rnn_cell_7/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????????????? : : : 2T
(simple_rnn_cell_7/BiasAdd/ReadVariableOp(simple_rnn_cell_7/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_7/MatMul/ReadVariableOp'simple_rnn_cell_7/MatMul/ReadVariableOp2V
)simple_rnn_cell_7/MatMul_1/ReadVariableOp)simple_rnn_cell_7/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?F
?
H__inference_simple_rnn_7_layer_call_and_return_conditional_losses_109162

inputsB
0simple_rnn_cell_7_matmul_readvariableop_resource: ?
1simple_rnn_cell_7_biasadd_readvariableop_resource:D
2simple_rnn_cell_7_matmul_1_readvariableop_resource:
identity??(simple_rnn_cell_7/BiasAdd/ReadVariableOp?'simple_rnn_cell_7/MatMul/ReadVariableOp?)simple_rnn_cell_7/MatMul_1/ReadVariableOp?while;
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
shrink_axis_maski
!simple_rnn_cell_7/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:f
!simple_rnn_cell_7/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
simple_rnn_cell_7/ones_likeFill*simple_rnn_cell_7/ones_like/Shape:output:0*simple_rnn_cell_7/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? a
#simple_rnn_cell_7/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:h
#simple_rnn_cell_7/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
simple_rnn_cell_7/ones_like_1Fill,simple_rnn_cell_7/ones_like_1/Shape:output:0,simple_rnn_cell_7/ones_like_1/Const:output:0*
T0*'
_output_shapes
:??????????
simple_rnn_cell_7/mulMulstrided_slice_2:output:0$simple_rnn_cell_7/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
'simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_7_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
simple_rnn_cell_7/MatMulMatMulsimple_rnn_cell_7/mul:z:0/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
(simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
simple_rnn_cell_7/BiasAddBiasAdd"simple_rnn_cell_7/MatMul:product:00simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
simple_rnn_cell_7/mul_1Mulzeros:output:0&simple_rnn_cell_7/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
)simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_7_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0?
simple_rnn_cell_7/MatMul_1MatMulsimple_rnn_cell_7/mul_1:z:01simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
simple_rnn_cell_7/addAddV2"simple_rnn_cell_7/BiasAdd:output:0$simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:?????????k
simple_rnn_cell_7/TanhTanhsimple_rnn_cell_7/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_7_matmul_readvariableop_resource1simple_rnn_cell_7_biasadd_readvariableop_resource2simple_rnn_cell_7_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_109088*
condR
while_cond_109087*8
output_shapes'
%: : : : :?????????: : : : : *
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
 :??????????????????g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp)^simple_rnn_cell_7/BiasAdd/ReadVariableOp(^simple_rnn_cell_7/MatMul/ReadVariableOp*^simple_rnn_cell_7/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????????????? : : : 2T
(simple_rnn_cell_7/BiasAdd/ReadVariableOp(simple_rnn_cell_7/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_7/MatMul/ReadVariableOp'simple_rnn_cell_7/MatMul/ReadVariableOp2V
)simple_rnn_cell_7/MatMul_1/ReadVariableOp)simple_rnn_cell_7/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?
?
H__inference_sequential_7_layer_call_and_return_conditional_losses_109212

inputs%
embedding_7_109036:	?' %
simple_rnn_7_109163: !
simple_rnn_7_109165:%
simple_rnn_7_109167:!
dense_14_109182:

dense_14_109184:
!
dense_15_109206:

dense_15_109208:
identity?? dense_14/StatefulPartitionedCall? dense_15/StatefulPartitionedCall?#embedding_7/StatefulPartitionedCall?$simple_rnn_7/StatefulPartitionedCall?
#embedding_7/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_7_109036*
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
G__inference_embedding_7_layer_call_and_return_conditional_losses_109035?
$simple_rnn_7/StatefulPartitionedCallStatefulPartitionedCall,embedding_7/StatefulPartitionedCall:output:0simple_rnn_7_109163simple_rnn_7_109165simple_rnn_7_109167*
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
GPU2*0J 8? *Q
fLRJ
H__inference_simple_rnn_7_layer_call_and_return_conditional_losses_109162?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_7/StatefulPartitionedCall:output:0dense_14_109182dense_14_109184*
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
D__inference_dense_14_layer_call_and_return_conditional_losses_109181?
dropout_7/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
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
E__inference_dropout_7_layer_call_and_return_conditional_losses_109192?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0dense_15_109206dense_15_109208*
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
D__inference_dense_15_layer_call_and_return_conditional_losses_109205x
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall$^embedding_7/StatefulPartitionedCall%^simple_rnn_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????????????: : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2J
#embedding_7/StatefulPartitionedCall#embedding_7/StatefulPartitionedCall2L
$simple_rnn_7/StatefulPartitionedCall$simple_rnn_7/StatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?	
?
G__inference_embedding_7_layer_call_and_return_conditional_losses_110015

inputs*
embedding_lookup_110009:	?' 
identity??embedding_lookup^
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:???????????????????
embedding_lookupResourceGatherembedding_lookup_110009Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/110009*4
_output_shapes"
 :?????????????????? *
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/110009*4
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
?
?
-__inference_simple_rnn_7_layer_call_fn_110059

inputs
unknown: 
	unknown_0:
	unknown_1:
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
GPU2*0J 8? *Q
fLRJ
H__inference_simple_rnn_7_layer_call_and_return_conditional_losses_109443o
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
?

?
D__inference_dense_14_layer_call_and_return_conditional_losses_109181

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
?5
?
while_body_110389
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_7_matmul_readvariableop_resource_0: G
9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0:L
:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_7_matmul_readvariableop_resource: E
7while_simple_rnn_cell_7_biasadd_readvariableop_resource:J
8while_simple_rnn_cell_7_matmul_1_readvariableop_resource:??.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_7/MatMul/ReadVariableOp?/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype0?
'while/simple_rnn_cell_7/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:l
'while/simple_rnn_cell_7/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
!while/simple_rnn_cell_7/ones_likeFill0while/simple_rnn_cell_7/ones_like/Shape:output:00while/simple_rnn_cell_7/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? l
)while/simple_rnn_cell_7/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:n
)while/simple_rnn_cell_7/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#while/simple_rnn_cell_7/ones_like_1Fill2while/simple_rnn_cell_7/ones_like_1/Shape:output:02while/simple_rnn_cell_7/ones_like_1/Const:output:0*
T0*'
_output_shapes
:??????????
while/simple_rnn_cell_7/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0*while/simple_rnn_cell_7/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
-while/simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_7_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0?
while/simple_rnn_cell_7/MatMulMatMulwhile/simple_rnn_cell_7/mul:z:05while/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
.while/simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/simple_rnn_cell_7/BiasAddBiasAdd(while/simple_rnn_cell_7/MatMul:product:06while/simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
while/simple_rnn_cell_7/mul_1Mulwhile_placeholder_2,while/simple_rnn_cell_7/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
/while/simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0?
 while/simple_rnn_cell_7/MatMul_1MatMul!while/simple_rnn_cell_7/mul_1:z:07while/simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
while/simple_rnn_cell_7/addAddV2(while/simple_rnn_cell_7/BiasAdd:output:0*while/simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:?????????w
while/simple_rnn_cell_7/TanhTanhwhile/simple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_7/Tanh:y:0*
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
: }
while/Identity_4Identity while/simple_rnn_cell_7/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp/^while/simple_rnn_cell_7/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_7/MatMul/ReadVariableOp0^while/simple_rnn_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_7_biasadd_readvariableop_resource9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_7_matmul_1_readvariableop_resource:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_7_matmul_readvariableop_resource8while_simple_rnn_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2`
.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_7/MatMul/ReadVariableOp-while/simple_rnn_cell_7/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp: 
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
:?????????:

_output_shapes
: :

_output_shapes
: 
?4
?
H__inference_simple_rnn_7_layer_call_and_return_conditional_losses_108826

inputs*
simple_rnn_cell_7_108751: &
simple_rnn_cell_7_108753:*
simple_rnn_cell_7_108755:
identity??)simple_rnn_cell_7/StatefulPartitionedCall?while;
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
)simple_rnn_cell_7/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_7_108751simple_rnn_cell_7_108753simple_rnn_cell_7_108755*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_simple_rnn_cell_7_layer_call_and_return_conditional_losses_108750n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_7_108751simple_rnn_cell_7_108753simple_rnn_cell_7_108755*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_108763*
condR
while_cond_108762*8
output_shapes'
%: : : : :?????????: : : : : *
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
 :??????????????????g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????z
NoOpNoOp*^simple_rnn_cell_7/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????????????? : : : 2V
)simple_rnn_cell_7/StatefulPartitionedCall)simple_rnn_cell_7/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?

?
2__inference_simple_rnn_cell_7_layer_call_fn_110700

inputs
states_0
unknown: 
	unknown_0:
	unknown_1:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_simple_rnn_cell_7_layer_call_and_return_conditional_losses_108750o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:????????? :?????????: : : 22
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
states/0
?
?
-__inference_simple_rnn_7_layer_call_fn_110037
inputs_0
unknown: 
	unknown_0:
	unknown_1:
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
GPU2*0J 8? *Q
fLRJ
H__inference_simple_rnn_7_layer_call_and_return_conditional_losses_109009o
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
?
?
)__inference_dense_14_layer_call_fn_110628

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
D__inference_dense_14_layer_call_and_return_conditional_losses_109181o
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
?J
?
while_body_109353
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_7_matmul_readvariableop_resource_0: G
9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0:L
:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_7_matmul_readvariableop_resource: E
7while_simple_rnn_cell_7_biasadd_readvariableop_resource:J
8while_simple_rnn_cell_7_matmul_1_readvariableop_resource:??.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_7/MatMul/ReadVariableOp?/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype0?
'while/simple_rnn_cell_7/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:l
'while/simple_rnn_cell_7/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
!while/simple_rnn_cell_7/ones_likeFill0while/simple_rnn_cell_7/ones_like/Shape:output:00while/simple_rnn_cell_7/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? j
%while/simple_rnn_cell_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#while/simple_rnn_cell_7/dropout/MulMul*while/simple_rnn_cell_7/ones_like:output:0.while/simple_rnn_cell_7/dropout/Const:output:0*
T0*'
_output_shapes
:????????? 
%while/simple_rnn_cell_7/dropout/ShapeShape*while/simple_rnn_cell_7/ones_like:output:0*
T0*
_output_shapes
:?
<while/simple_rnn_cell_7/dropout/random_uniform/RandomUniformRandomUniform.while/simple_rnn_cell_7/dropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???s
.while/simple_rnn_cell_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
,while/simple_rnn_cell_7/dropout/GreaterEqualGreaterEqualEwhile/simple_rnn_cell_7/dropout/random_uniform/RandomUniform:output:07while/simple_rnn_cell_7/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
$while/simple_rnn_cell_7/dropout/CastCast0while/simple_rnn_cell_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
%while/simple_rnn_cell_7/dropout/Mul_1Mul'while/simple_rnn_cell_7/dropout/Mul:z:0(while/simple_rnn_cell_7/dropout/Cast:y:0*
T0*'
_output_shapes
:????????? l
)while/simple_rnn_cell_7/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:n
)while/simple_rnn_cell_7/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#while/simple_rnn_cell_7/ones_like_1Fill2while/simple_rnn_cell_7/ones_like_1/Shape:output:02while/simple_rnn_cell_7/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????l
'while/simple_rnn_cell_7/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
%while/simple_rnn_cell_7/dropout_1/MulMul,while/simple_rnn_cell_7/ones_like_1:output:00while/simple_rnn_cell_7/dropout_1/Const:output:0*
T0*'
_output_shapes
:??????????
'while/simple_rnn_cell_7/dropout_1/ShapeShape,while/simple_rnn_cell_7/ones_like_1:output:0*
T0*
_output_shapes
:?
>while/simple_rnn_cell_7/dropout_1/random_uniform/RandomUniformRandomUniform0while/simple_rnn_cell_7/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???u
0while/simple_rnn_cell_7/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
.while/simple_rnn_cell_7/dropout_1/GreaterEqualGreaterEqualGwhile/simple_rnn_cell_7/dropout_1/random_uniform/RandomUniform:output:09while/simple_rnn_cell_7/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
&while/simple_rnn_cell_7/dropout_1/CastCast2while/simple_rnn_cell_7/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
'while/simple_rnn_cell_7/dropout_1/Mul_1Mul)while/simple_rnn_cell_7/dropout_1/Mul:z:0*while/simple_rnn_cell_7/dropout_1/Cast:y:0*
T0*'
_output_shapes
:??????????
while/simple_rnn_cell_7/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/simple_rnn_cell_7/dropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
-while/simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_7_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0?
while/simple_rnn_cell_7/MatMulMatMulwhile/simple_rnn_cell_7/mul:z:05while/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
.while/simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/simple_rnn_cell_7/BiasAddBiasAdd(while/simple_rnn_cell_7/MatMul:product:06while/simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
while/simple_rnn_cell_7/mul_1Mulwhile_placeholder_2+while/simple_rnn_cell_7/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:??????????
/while/simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0?
 while/simple_rnn_cell_7/MatMul_1MatMul!while/simple_rnn_cell_7/mul_1:z:07while/simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
while/simple_rnn_cell_7/addAddV2(while/simple_rnn_cell_7/BiasAdd:output:0*while/simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:?????????w
while/simple_rnn_cell_7/TanhTanhwhile/simple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_7/Tanh:y:0*
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
: }
while/Identity_4Identity while/simple_rnn_cell_7/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp/^while/simple_rnn_cell_7/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_7/MatMul/ReadVariableOp0^while/simple_rnn_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_7_biasadd_readvariableop_resource9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_7_matmul_1_readvariableop_resource:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_7_matmul_readvariableop_resource8while_simple_rnn_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2`
.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_7/MatMul/ReadVariableOp-while/simple_rnn_cell_7/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp: 
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
:?????????:

_output_shapes
: :

_output_shapes
: 
?	
?
-__inference_sequential_7_layer_call_fn_109669

inputs
unknown:	?' 
	unknown_0: 
	unknown_1:
	unknown_2:
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
H__inference_sequential_7_layer_call_and_return_conditional_losses_109508o
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
?$
?
M__inference_simple_rnn_cell_7_layer_call_and_return_conditional_losses_110780

inputs
states_00
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:2
 matmul_1_readvariableop_resource:
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpE
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
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??v
dropout_1/MulMulones_like_1:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????S
dropout_1/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
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
:?????????s
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????o
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????W
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? t
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0j
MatMulMatMulmul:z:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????]
mul_1Mulstates_0dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0p
MatMul_1MatMul	mul_1:z:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????G
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:????????? :?????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0
?
?
while_cond_110248
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_110248___redundant_placeholder04
0while_while_cond_110248___redundant_placeholder14
0while_while_cond_110248___redundant_placeholder24
0while_while_cond_110248___redundant_placeholder3
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
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????: ::::: 
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
:?????????:

_output_shapes
: :

_output_shapes
:
?$
?
M__inference_simple_rnn_cell_7_layer_call_and_return_conditional_losses_108894

inputs

states0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:2
 matmul_1_readvariableop_resource:
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpE
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
seed???)*
seed2??[
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
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??v
dropout_1/MulMulones_like_1:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????S
dropout_1/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2?̇]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????s
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????o
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????W
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? t
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0j
MatMulMatMulmul:z:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????[
mul_1Mulstatesdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0p
MatMul_1MatMul	mul_1:z:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????G
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:????????? :?????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_namestates
?
?
H__inference_sequential_7_layer_call_and_return_conditional_losses_109573
embedding_7_input%
embedding_7_109551:	?' %
simple_rnn_7_109554: !
simple_rnn_7_109556:%
simple_rnn_7_109558:!
dense_14_109561:

dense_14_109563:
!
dense_15_109567:

dense_15_109569:
identity?? dense_14/StatefulPartitionedCall? dense_15/StatefulPartitionedCall?#embedding_7/StatefulPartitionedCall?$simple_rnn_7/StatefulPartitionedCall?
#embedding_7/StatefulPartitionedCallStatefulPartitionedCallembedding_7_inputembedding_7_109551*
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
G__inference_embedding_7_layer_call_and_return_conditional_losses_109035?
$simple_rnn_7/StatefulPartitionedCallStatefulPartitionedCall,embedding_7/StatefulPartitionedCall:output:0simple_rnn_7_109554simple_rnn_7_109556simple_rnn_7_109558*
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
GPU2*0J 8? *Q
fLRJ
H__inference_simple_rnn_7_layer_call_and_return_conditional_losses_109162?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_7/StatefulPartitionedCall:output:0dense_14_109561dense_14_109563*
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
D__inference_dense_14_layer_call_and_return_conditional_losses_109181?
dropout_7/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
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
E__inference_dropout_7_layer_call_and_return_conditional_losses_109192?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0dense_15_109567dense_15_109569*
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
D__inference_dense_15_layer_call_and_return_conditional_losses_109205x
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall$^embedding_7/StatefulPartitionedCall%^simple_rnn_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????????????: : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2J
#embedding_7/StatefulPartitionedCall#embedding_7/StatefulPartitionedCall2L
$simple_rnn_7/StatefulPartitionedCall$simple_rnn_7/StatefulPartitionedCall:c _
0
_output_shapes
:??????????????????
+
_user_specified_nameembedding_7_input
?	
d
E__inference_dropout_7_layer_call_and_return_conditional_losses_110666

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
?
c
E__inference_dropout_7_layer_call_and_return_conditional_losses_110654

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
?m
?
H__inference_sequential_7_layer_call_and_return_conditional_losses_109814

inputs6
#embedding_7_embedding_lookup_109673:	?' O
=simple_rnn_7_simple_rnn_cell_7_matmul_readvariableop_resource: L
>simple_rnn_7_simple_rnn_cell_7_biasadd_readvariableop_resource:Q
?simple_rnn_7_simple_rnn_cell_7_matmul_1_readvariableop_resource:9
'dense_14_matmul_readvariableop_resource:
6
(dense_14_biasadd_readvariableop_resource:
9
'dense_15_matmul_readvariableop_resource:
6
(dense_15_biasadd_readvariableop_resource:
identity??dense_14/BiasAdd/ReadVariableOp?dense_14/MatMul/ReadVariableOp?dense_15/BiasAdd/ReadVariableOp?dense_15/MatMul/ReadVariableOp?embedding_7/embedding_lookup?5simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOp?4simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOp?6simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOp?simple_rnn_7/whilej
embedding_7/CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:???????????????????
embedding_7/embedding_lookupResourceGather#embedding_7_embedding_lookup_109673embedding_7/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding_7/embedding_lookup/109673*4
_output_shapes"
 :?????????????????? *
dtype0?
%embedding_7/embedding_lookup/IdentityIdentity%embedding_7/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_7/embedding_lookup/109673*4
_output_shapes"
 :?????????????????? ?
'embedding_7/embedding_lookup/Identity_1Identity.embedding_7/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :?????????????????? r
simple_rnn_7/ShapeShape0embedding_7/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:j
 simple_rnn_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_7/strided_sliceStridedSlicesimple_rnn_7/Shape:output:0)simple_rnn_7/strided_slice/stack:output:0+simple_rnn_7/strided_slice/stack_1:output:0+simple_rnn_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_7/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_7/zeros/packedPack#simple_rnn_7/strided_slice:output:0$simple_rnn_7/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_7/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
simple_rnn_7/zerosFill"simple_rnn_7/zeros/packed:output:0!simple_rnn_7/zeros/Const:output:0*
T0*'
_output_shapes
:?????????p
simple_rnn_7/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_7/transpose	Transpose0embedding_7/embedding_lookup/Identity_1:output:0$simple_rnn_7/transpose/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? ^
simple_rnn_7/Shape_1Shapesimple_rnn_7/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_7/strided_slice_1StridedSlicesimple_rnn_7/Shape_1:output:0+simple_rnn_7/strided_slice_1/stack:output:0-simple_rnn_7/strided_slice_1/stack_1:output:0-simple_rnn_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_7/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
simple_rnn_7/TensorArrayV2TensorListReserve1simple_rnn_7/TensorArrayV2/element_shape:output:0%simple_rnn_7/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Bsimple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
4simple_rnn_7/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_7/transpose:y:0Ksimple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???l
"simple_rnn_7/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_7/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_7/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_7/strided_slice_2StridedSlicesimple_rnn_7/transpose:y:0+simple_rnn_7/strided_slice_2/stack:output:0-simple_rnn_7/strided_slice_2/stack_1:output:0-simple_rnn_7/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask?
.simple_rnn_7/simple_rnn_cell_7/ones_like/ShapeShape%simple_rnn_7/strided_slice_2:output:0*
T0*
_output_shapes
:s
.simple_rnn_7/simple_rnn_cell_7/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
(simple_rnn_7/simple_rnn_cell_7/ones_likeFill7simple_rnn_7/simple_rnn_cell_7/ones_like/Shape:output:07simple_rnn_7/simple_rnn_cell_7/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? {
0simple_rnn_7/simple_rnn_cell_7/ones_like_1/ShapeShapesimple_rnn_7/zeros:output:0*
T0*
_output_shapes
:u
0simple_rnn_7/simple_rnn_cell_7/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
*simple_rnn_7/simple_rnn_cell_7/ones_like_1Fill9simple_rnn_7/simple_rnn_cell_7/ones_like_1/Shape:output:09simple_rnn_7/simple_rnn_cell_7/ones_like_1/Const:output:0*
T0*'
_output_shapes
:??????????
"simple_rnn_7/simple_rnn_cell_7/mulMul%simple_rnn_7/strided_slice_2:output:01simple_rnn_7/simple_rnn_cell_7/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
4simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOp=simple_rnn_7_simple_rnn_cell_7_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
%simple_rnn_7/simple_rnn_cell_7/MatMulMatMul&simple_rnn_7/simple_rnn_cell_7/mul:z:0<simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
5simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_7_simple_rnn_cell_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
&simple_rnn_7/simple_rnn_cell_7/BiasAddBiasAdd/simple_rnn_7/simple_rnn_cell_7/MatMul:product:0=simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
$simple_rnn_7/simple_rnn_cell_7/mul_1Mulsimple_rnn_7/zeros:output:03simple_rnn_7/simple_rnn_cell_7/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
6simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_7_simple_rnn_cell_7_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0?
'simple_rnn_7/simple_rnn_cell_7/MatMul_1MatMul(simple_rnn_7/simple_rnn_cell_7/mul_1:z:0>simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
"simple_rnn_7/simple_rnn_cell_7/addAddV2/simple_rnn_7/simple_rnn_cell_7/BiasAdd:output:01simple_rnn_7/simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:??????????
#simple_rnn_7/simple_rnn_cell_7/TanhTanh&simple_rnn_7/simple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:?????????{
*simple_rnn_7/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
simple_rnn_7/TensorArrayV2_1TensorListReserve3simple_rnn_7/TensorArrayV2_1/element_shape:output:0%simple_rnn_7/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???S
simple_rnn_7/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_7/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????a
simple_rnn_7/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
simple_rnn_7/whileWhile(simple_rnn_7/while/loop_counter:output:0.simple_rnn_7/while/maximum_iterations:output:0simple_rnn_7/time:output:0%simple_rnn_7/TensorArrayV2_1:handle:0simple_rnn_7/zeros:output:0%simple_rnn_7/strided_slice_1:output:0Dsimple_rnn_7/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_7_simple_rnn_cell_7_matmul_readvariableop_resource>simple_rnn_7_simple_rnn_cell_7_biasadd_readvariableop_resource?simple_rnn_7_simple_rnn_cell_7_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( **
body"R 
simple_rnn_7_while_body_109725**
cond"R 
simple_rnn_7_while_cond_109724*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations ?
=simple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
/simple_rnn_7/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_7/while:output:3Fsimple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype0u
"simple_rnn_7/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????n
$simple_rnn_7/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_7/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_7/strided_slice_3StridedSlice8simple_rnn_7/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_7/strided_slice_3/stack:output:0-simple_rnn_7/strided_slice_3/stack_1:output:0-simple_rnn_7/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskr
simple_rnn_7/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_7/transpose_1	Transpose8simple_rnn_7/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_7/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :???????????????????
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
dense_14/MatMulMatMul%simple_rnn_7/strided_slice_3:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
b
dense_14/ReluReludense_14/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
m
dropout_7/IdentityIdentitydense_14/Relu:activations:0*
T0*'
_output_shapes
:?????????
?
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
dense_15/MatMulMatMuldropout_7/Identity:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_15/SigmoidSigmoiddense_15/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
IdentityIdentitydense_15/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp^embedding_7/embedding_lookup6^simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOp5^simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOp7^simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOp^simple_rnn_7/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????????????: : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2<
embedding_7/embedding_lookupembedding_7/embedding_lookup2n
5simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOp5simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOp2l
4simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOp4simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOp2p
6simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOp6simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOp2(
simple_rnn_7/whilesimple_rnn_7/while:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
while_cond_109087
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_109087___redundant_placeholder04
0while_while_cond_109087___redundant_placeholder14
0while_while_cond_109087___redundant_placeholder24
0while_while_cond_109087___redundant_placeholder3
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
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????: ::::: 
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
:?????????:

_output_shapes
: :

_output_shapes
:
?5
?
while_body_110109
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_7_matmul_readvariableop_resource_0: G
9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0:L
:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_7_matmul_readvariableop_resource: E
7while_simple_rnn_cell_7_biasadd_readvariableop_resource:J
8while_simple_rnn_cell_7_matmul_1_readvariableop_resource:??.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_7/MatMul/ReadVariableOp?/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype0?
'while/simple_rnn_cell_7/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:l
'while/simple_rnn_cell_7/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
!while/simple_rnn_cell_7/ones_likeFill0while/simple_rnn_cell_7/ones_like/Shape:output:00while/simple_rnn_cell_7/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? l
)while/simple_rnn_cell_7/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:n
)while/simple_rnn_cell_7/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#while/simple_rnn_cell_7/ones_like_1Fill2while/simple_rnn_cell_7/ones_like_1/Shape:output:02while/simple_rnn_cell_7/ones_like_1/Const:output:0*
T0*'
_output_shapes
:??????????
while/simple_rnn_cell_7/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0*while/simple_rnn_cell_7/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
-while/simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_7_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0?
while/simple_rnn_cell_7/MatMulMatMulwhile/simple_rnn_cell_7/mul:z:05while/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
.while/simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/simple_rnn_cell_7/BiasAddBiasAdd(while/simple_rnn_cell_7/MatMul:product:06while/simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
while/simple_rnn_cell_7/mul_1Mulwhile_placeholder_2,while/simple_rnn_cell_7/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
/while/simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0?
 while/simple_rnn_cell_7/MatMul_1MatMul!while/simple_rnn_cell_7/mul_1:z:07while/simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
while/simple_rnn_cell_7/addAddV2(while/simple_rnn_cell_7/BiasAdd:output:0*while/simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:?????????w
while/simple_rnn_cell_7/TanhTanhwhile/simple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_7/Tanh:y:0*
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
: }
while/Identity_4Identity while/simple_rnn_cell_7/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp/^while/simple_rnn_cell_7/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_7/MatMul/ReadVariableOp0^while/simple_rnn_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_7_biasadd_readvariableop_resource9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_7_matmul_1_readvariableop_resource:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_7_matmul_readvariableop_resource8while_simple_rnn_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2`
.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_7/MatMul/ReadVariableOp-while/simple_rnn_cell_7/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp: 
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
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
M__inference_simple_rnn_cell_7_layer_call_and_return_conditional_losses_110739

inputs
states_00
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:2
 matmul_1_readvariableop_resource:
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpE
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
:????????? t
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0j
MatMulMatMulmul:z:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^
mul_1Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:?????????x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0p
MatMul_1MatMul	mul_1:z:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????G
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:????????? :?????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0
?

?
simple_rnn_7_while_cond_1097246
2simple_rnn_7_while_simple_rnn_7_while_loop_counter<
8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations"
simple_rnn_7_while_placeholder$
 simple_rnn_7_while_placeholder_1$
 simple_rnn_7_while_placeholder_28
4simple_rnn_7_while_less_simple_rnn_7_strided_slice_1N
Jsimple_rnn_7_while_simple_rnn_7_while_cond_109724___redundant_placeholder0N
Jsimple_rnn_7_while_simple_rnn_7_while_cond_109724___redundant_placeholder1N
Jsimple_rnn_7_while_simple_rnn_7_while_cond_109724___redundant_placeholder2N
Jsimple_rnn_7_while_simple_rnn_7_while_cond_109724___redundant_placeholder3
simple_rnn_7_while_identity
?
simple_rnn_7/while/LessLesssimple_rnn_7_while_placeholder4simple_rnn_7_while_less_simple_rnn_7_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_7/while/IdentityIdentitysimple_rnn_7/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_7_while_identity$simple_rnn_7/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????: ::::: 
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
:?????????:

_output_shapes
: :

_output_shapes
:
?F
?
H__inference_simple_rnn_7_layer_call_and_return_conditional_losses_110183
inputs_0B
0simple_rnn_cell_7_matmul_readvariableop_resource: ?
1simple_rnn_cell_7_biasadd_readvariableop_resource:D
2simple_rnn_cell_7_matmul_1_readvariableop_resource:
identity??(simple_rnn_cell_7/BiasAdd/ReadVariableOp?'simple_rnn_cell_7/MatMul/ReadVariableOp?)simple_rnn_cell_7/MatMul_1/ReadVariableOp?while=
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
shrink_axis_maski
!simple_rnn_cell_7/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:f
!simple_rnn_cell_7/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
simple_rnn_cell_7/ones_likeFill*simple_rnn_cell_7/ones_like/Shape:output:0*simple_rnn_cell_7/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? a
#simple_rnn_cell_7/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:h
#simple_rnn_cell_7/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
simple_rnn_cell_7/ones_like_1Fill,simple_rnn_cell_7/ones_like_1/Shape:output:0,simple_rnn_cell_7/ones_like_1/Const:output:0*
T0*'
_output_shapes
:??????????
simple_rnn_cell_7/mulMulstrided_slice_2:output:0$simple_rnn_cell_7/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
'simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_7_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
simple_rnn_cell_7/MatMulMatMulsimple_rnn_cell_7/mul:z:0/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
(simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
simple_rnn_cell_7/BiasAddBiasAdd"simple_rnn_cell_7/MatMul:product:00simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
simple_rnn_cell_7/mul_1Mulzeros:output:0&simple_rnn_cell_7/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
)simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_7_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0?
simple_rnn_cell_7/MatMul_1MatMulsimple_rnn_cell_7/mul_1:z:01simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
simple_rnn_cell_7/addAddV2"simple_rnn_cell_7/BiasAdd:output:0$simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:?????????k
simple_rnn_cell_7/TanhTanhsimple_rnn_cell_7/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_7_matmul_readvariableop_resource1simple_rnn_cell_7_biasadd_readvariableop_resource2simple_rnn_cell_7_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_110109*
condR
while_cond_110108*8
output_shapes'
%: : : : :?????????: : : : : *
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
 :??????????????????g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp)^simple_rnn_cell_7/BiasAdd/ReadVariableOp(^simple_rnn_cell_7/MatMul/ReadVariableOp*^simple_rnn_cell_7/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????????????? : : : 2T
(simple_rnn_cell_7/BiasAdd/ReadVariableOp(simple_rnn_cell_7/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_7/MatMul/ReadVariableOp'simple_rnn_cell_7/MatMul/ReadVariableOp2V
)simple_rnn_cell_7/MatMul_1/ReadVariableOp)simple_rnn_cell_7/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :?????????????????? 
"
_user_specified_name
inputs/0
?\
?
simple_rnn_7_while_body_1098866
2simple_rnn_7_while_simple_rnn_7_while_loop_counter<
8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations"
simple_rnn_7_while_placeholder$
 simple_rnn_7_while_placeholder_1$
 simple_rnn_7_while_placeholder_25
1simple_rnn_7_while_simple_rnn_7_strided_slice_1_0q
msimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0W
Esimple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resource_0: T
Fsimple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resource_0:Y
Gsimple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0:
simple_rnn_7_while_identity!
simple_rnn_7_while_identity_1!
simple_rnn_7_while_identity_2!
simple_rnn_7_while_identity_3!
simple_rnn_7_while_identity_43
/simple_rnn_7_while_simple_rnn_7_strided_slice_1o
ksimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensorU
Csimple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resource: R
Dsimple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resource:W
Esimple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resource:??;simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOp?:simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOp?<simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp?
Dsimple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
6simple_rnn_7/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_7_while_placeholderMsimple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype0?
4simple_rnn_7/while/simple_rnn_cell_7/ones_like/ShapeShape=simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:y
4simple_rnn_7/while/simple_rnn_cell_7/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
.simple_rnn_7/while/simple_rnn_cell_7/ones_likeFill=simple_rnn_7/while/simple_rnn_cell_7/ones_like/Shape:output:0=simple_rnn_7/while/simple_rnn_cell_7/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? w
2simple_rnn_7/while/simple_rnn_cell_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
0simple_rnn_7/while/simple_rnn_cell_7/dropout/MulMul7simple_rnn_7/while/simple_rnn_cell_7/ones_like:output:0;simple_rnn_7/while/simple_rnn_cell_7/dropout/Const:output:0*
T0*'
_output_shapes
:????????? ?
2simple_rnn_7/while/simple_rnn_cell_7/dropout/ShapeShape7simple_rnn_7/while/simple_rnn_cell_7/ones_like:output:0*
T0*
_output_shapes
:?
Isimple_rnn_7/while/simple_rnn_cell_7/dropout/random_uniform/RandomUniformRandomUniform;simple_rnn_7/while/simple_rnn_cell_7/dropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2????
;simple_rnn_7/while/simple_rnn_cell_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
9simple_rnn_7/while/simple_rnn_cell_7/dropout/GreaterEqualGreaterEqualRsimple_rnn_7/while/simple_rnn_cell_7/dropout/random_uniform/RandomUniform:output:0Dsimple_rnn_7/while/simple_rnn_cell_7/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
1simple_rnn_7/while/simple_rnn_cell_7/dropout/CastCast=simple_rnn_7/while/simple_rnn_cell_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
2simple_rnn_7/while/simple_rnn_cell_7/dropout/Mul_1Mul4simple_rnn_7/while/simple_rnn_cell_7/dropout/Mul:z:05simple_rnn_7/while/simple_rnn_cell_7/dropout/Cast:y:0*
T0*'
_output_shapes
:????????? ?
6simple_rnn_7/while/simple_rnn_cell_7/ones_like_1/ShapeShape simple_rnn_7_while_placeholder_2*
T0*
_output_shapes
:{
6simple_rnn_7/while/simple_rnn_cell_7/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
0simple_rnn_7/while/simple_rnn_cell_7/ones_like_1Fill?simple_rnn_7/while/simple_rnn_cell_7/ones_like_1/Shape:output:0?simple_rnn_7/while/simple_rnn_cell_7/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????y
4simple_rnn_7/while/simple_rnn_cell_7/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
2simple_rnn_7/while/simple_rnn_cell_7/dropout_1/MulMul9simple_rnn_7/while/simple_rnn_cell_7/ones_like_1:output:0=simple_rnn_7/while/simple_rnn_cell_7/dropout_1/Const:output:0*
T0*'
_output_shapes
:??????????
4simple_rnn_7/while/simple_rnn_cell_7/dropout_1/ShapeShape9simple_rnn_7/while/simple_rnn_cell_7/ones_like_1:output:0*
T0*
_output_shapes
:?
Ksimple_rnn_7/while/simple_rnn_cell_7/dropout_1/random_uniform/RandomUniformRandomUniform=simple_rnn_7/while/simple_rnn_cell_7/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2????
=simple_rnn_7/while/simple_rnn_cell_7/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
;simple_rnn_7/while/simple_rnn_cell_7/dropout_1/GreaterEqualGreaterEqualTsimple_rnn_7/while/simple_rnn_cell_7/dropout_1/random_uniform/RandomUniform:output:0Fsimple_rnn_7/while/simple_rnn_cell_7/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
3simple_rnn_7/while/simple_rnn_cell_7/dropout_1/CastCast?simple_rnn_7/while/simple_rnn_cell_7/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
4simple_rnn_7/while/simple_rnn_cell_7/dropout_1/Mul_1Mul6simple_rnn_7/while/simple_rnn_cell_7/dropout_1/Mul:z:07simple_rnn_7/while/simple_rnn_cell_7/dropout_1/Cast:y:0*
T0*'
_output_shapes
:??????????
(simple_rnn_7/while/simple_rnn_cell_7/mulMul=simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem:item:06simple_rnn_7/while/simple_rnn_cell_7/dropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
:simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0?
+simple_rnn_7/while/simple_rnn_cell_7/MatMulMatMul,simple_rnn_7/while/simple_rnn_cell_7/mul:z:0Bsimple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
;simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
,simple_rnn_7/while/simple_rnn_cell_7/BiasAddBiasAdd5simple_rnn_7/while/simple_rnn_cell_7/MatMul:product:0Csimple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*simple_rnn_7/while/simple_rnn_cell_7/mul_1Mul simple_rnn_7_while_placeholder_28simple_rnn_7/while/simple_rnn_cell_7/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:??????????
<simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0?
-simple_rnn_7/while/simple_rnn_cell_7/MatMul_1MatMul.simple_rnn_7/while/simple_rnn_cell_7/mul_1:z:0Dsimple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
(simple_rnn_7/while/simple_rnn_cell_7/addAddV25simple_rnn_7/while/simple_rnn_cell_7/BiasAdd:output:07simple_rnn_7/while/simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:??????????
)simple_rnn_7/while/simple_rnn_cell_7/TanhTanh,simple_rnn_7/while/simple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:??????????
7simple_rnn_7/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_7_while_placeholder_1simple_rnn_7_while_placeholder-simple_rnn_7/while/simple_rnn_cell_7/Tanh:y:0*
_output_shapes
: *
element_dtype0:???Z
simple_rnn_7/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_7/while/addAddV2simple_rnn_7_while_placeholder!simple_rnn_7/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_7/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_7/while/add_1AddV22simple_rnn_7_while_simple_rnn_7_while_loop_counter#simple_rnn_7/while/add_1/y:output:0*
T0*
_output_shapes
: ?
simple_rnn_7/while/IdentityIdentitysimple_rnn_7/while/add_1:z:0^simple_rnn_7/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_7/while/Identity_1Identity8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations^simple_rnn_7/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_7/while/Identity_2Identitysimple_rnn_7/while/add:z:0^simple_rnn_7/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_7/while/Identity_3IdentityGsimple_rnn_7/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_7/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_7/while/Identity_4Identity-simple_rnn_7/while/simple_rnn_cell_7/Tanh:y:0^simple_rnn_7/while/NoOp*
T0*'
_output_shapes
:??????????
simple_rnn_7/while/NoOpNoOp<^simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOp;^simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOp=^simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_7_while_identity$simple_rnn_7/while/Identity:output:0"G
simple_rnn_7_while_identity_1&simple_rnn_7/while/Identity_1:output:0"G
simple_rnn_7_while_identity_2&simple_rnn_7/while/Identity_2:output:0"G
simple_rnn_7_while_identity_3&simple_rnn_7/while/Identity_3:output:0"G
simple_rnn_7_while_identity_4&simple_rnn_7/while/Identity_4:output:0"d
/simple_rnn_7_while_simple_rnn_7_strided_slice_11simple_rnn_7_while_simple_rnn_7_strided_slice_1_0"?
Dsimple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resourceFsimple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resource_0"?
Esimple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resourceGsimple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0"?
Csimple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resourceEsimple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resource_0"?
ksimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensormsimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2z
;simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOp;simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOp2x
:simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOp:simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOp2|
<simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp<simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp: 
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
:?????????:

_output_shapes
: :

_output_shapes
: 
?	
d
E__inference_dropout_7_layer_call_and_return_conditional_losses_109261

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
?
?
while_cond_108762
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_108762___redundant_placeholder04
0while_while_cond_108762___redundant_placeholder14
0while_while_cond_108762___redundant_placeholder24
0while_while_cond_108762___redundant_placeholder3
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
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????: ::::: 
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
:?????????:

_output_shapes
: :

_output_shapes
:
?	
?
-__inference_sequential_7_layer_call_fn_109648

inputs
unknown:	?' 
	unknown_0: 
	unknown_1:
	unknown_2:
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
H__inference_sequential_7_layer_call_and_return_conditional_losses_109212o
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
?
?
,__inference_embedding_7_layer_call_fn_110005

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
G__inference_embedding_7_layer_call_and_return_conditional_losses_109035|
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
?4
?
H__inference_simple_rnn_7_layer_call_and_return_conditional_losses_109009

inputs*
simple_rnn_cell_7_108934: &
simple_rnn_cell_7_108936:*
simple_rnn_cell_7_108938:
identity??)simple_rnn_cell_7/StatefulPartitionedCall?while;
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
)simple_rnn_cell_7/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_7_108934simple_rnn_cell_7_108936simple_rnn_cell_7_108938*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_simple_rnn_cell_7_layer_call_and_return_conditional_losses_108894n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_7_108934simple_rnn_cell_7_108936simple_rnn_cell_7_108938*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_108946*
condR
while_cond_108945*8
output_shapes'
%: : : : :?????????: : : : : *
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
 :??????????????????g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????z
NoOpNoOp*^simple_rnn_cell_7/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????????????? : : : 2V
)simple_rnn_cell_7/StatefulPartitionedCall)simple_rnn_cell_7/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?

?
2__inference_simple_rnn_cell_7_layer_call_fn_110714

inputs
states_0
unknown: 
	unknown_0:
	unknown_1:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_simple_rnn_cell_7_layer_call_and_return_conditional_losses_108894o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:????????? :?????????: : : 22
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
states/0
?5
?
while_body_109088
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_7_matmul_readvariableop_resource_0: G
9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0:L
:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_7_matmul_readvariableop_resource: E
7while_simple_rnn_cell_7_biasadd_readvariableop_resource:J
8while_simple_rnn_cell_7_matmul_1_readvariableop_resource:??.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_7/MatMul/ReadVariableOp?/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype0?
'while/simple_rnn_cell_7/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:l
'while/simple_rnn_cell_7/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
!while/simple_rnn_cell_7/ones_likeFill0while/simple_rnn_cell_7/ones_like/Shape:output:00while/simple_rnn_cell_7/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? l
)while/simple_rnn_cell_7/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:n
)while/simple_rnn_cell_7/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#while/simple_rnn_cell_7/ones_like_1Fill2while/simple_rnn_cell_7/ones_like_1/Shape:output:02while/simple_rnn_cell_7/ones_like_1/Const:output:0*
T0*'
_output_shapes
:??????????
while/simple_rnn_cell_7/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0*while/simple_rnn_cell_7/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
-while/simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_7_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0?
while/simple_rnn_cell_7/MatMulMatMulwhile/simple_rnn_cell_7/mul:z:05while/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
.while/simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/simple_rnn_cell_7/BiasAddBiasAdd(while/simple_rnn_cell_7/MatMul:product:06while/simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
while/simple_rnn_cell_7/mul_1Mulwhile_placeholder_2,while/simple_rnn_cell_7/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
/while/simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0?
 while/simple_rnn_cell_7/MatMul_1MatMul!while/simple_rnn_cell_7/mul_1:z:07while/simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
while/simple_rnn_cell_7/addAddV2(while/simple_rnn_cell_7/BiasAdd:output:0*while/simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:?????????w
while/simple_rnn_cell_7/TanhTanhwhile/simple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_7/Tanh:y:0*
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
: }
while/Identity_4Identity while/simple_rnn_cell_7/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp/^while/simple_rnn_cell_7/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_7/MatMul/ReadVariableOp0^while/simple_rnn_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_7_biasadd_readvariableop_resource9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_7_matmul_1_readvariableop_resource:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_7_matmul_readvariableop_resource8while_simple_rnn_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2`
.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_7/MatMul/ReadVariableOp-while/simple_rnn_cell_7/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp: 
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
:?????????:

_output_shapes
: :

_output_shapes
: 
? 
?
while_body_108946
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
 while_simple_rnn_cell_7_108968_0: .
 while_simple_rnn_cell_7_108970_0:2
 while_simple_rnn_cell_7_108972_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_simple_rnn_cell_7_108968: ,
while_simple_rnn_cell_7_108970:0
while_simple_rnn_cell_7_108972:??/while/simple_rnn_cell_7/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype0?
/while/simple_rnn_cell_7/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2 while_simple_rnn_cell_7_108968_0 while_simple_rnn_cell_7_108970_0 while_simple_rnn_cell_7_108972_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_simple_rnn_cell_7_layer_call_and_return_conditional_losses_108894?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_7/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity8while/simple_rnn_cell_7/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????~

while/NoOpNoOp0^while/simple_rnn_cell_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"B
while_simple_rnn_cell_7_108968 while_simple_rnn_cell_7_108968_0"B
while_simple_rnn_cell_7_108970 while_simple_rnn_cell_7_108970_0"B
while_simple_rnn_cell_7_108972 while_simple_rnn_cell_7_108972_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2b
/while/simple_rnn_cell_7/StatefulPartitionedCall/while/simple_rnn_cell_7/StatefulPartitionedCall: 
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
:?????????:

_output_shapes
: :

_output_shapes
: 
? 
?
while_body_108763
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
 while_simple_rnn_cell_7_108785_0: .
 while_simple_rnn_cell_7_108787_0:2
 while_simple_rnn_cell_7_108789_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_simple_rnn_cell_7_108785: ,
while_simple_rnn_cell_7_108787:0
while_simple_rnn_cell_7_108789:??/while/simple_rnn_cell_7/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype0?
/while/simple_rnn_cell_7/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2 while_simple_rnn_cell_7_108785_0 while_simple_rnn_cell_7_108787_0 while_simple_rnn_cell_7_108789_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_simple_rnn_cell_7_layer_call_and_return_conditional_losses_108750?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_7/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity8while/simple_rnn_cell_7/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????~

while/NoOpNoOp0^while/simple_rnn_cell_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"B
while_simple_rnn_cell_7_108785 while_simple_rnn_cell_7_108785_0"B
while_simple_rnn_cell_7_108787 while_simple_rnn_cell_7_108787_0"B
while_simple_rnn_cell_7_108789 while_simple_rnn_cell_7_108789_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2b
/while/simple_rnn_cell_7/StatefulPartitionedCall/while/simple_rnn_cell_7/StatefulPartitionedCall: 
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
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_110108
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_110108___redundant_placeholder04
0while_while_cond_110108___redundant_placeholder14
0while_while_cond_110108___redundant_placeholder24
0while_while_cond_110108___redundant_placeholder3
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
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????: ::::: 
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
:?????????:

_output_shapes
: :

_output_shapes
:
?C
?
simple_rnn_7_while_body_1097256
2simple_rnn_7_while_simple_rnn_7_while_loop_counter<
8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations"
simple_rnn_7_while_placeholder$
 simple_rnn_7_while_placeholder_1$
 simple_rnn_7_while_placeholder_25
1simple_rnn_7_while_simple_rnn_7_strided_slice_1_0q
msimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0W
Esimple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resource_0: T
Fsimple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resource_0:Y
Gsimple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0:
simple_rnn_7_while_identity!
simple_rnn_7_while_identity_1!
simple_rnn_7_while_identity_2!
simple_rnn_7_while_identity_3!
simple_rnn_7_while_identity_43
/simple_rnn_7_while_simple_rnn_7_strided_slice_1o
ksimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensorU
Csimple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resource: R
Dsimple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resource:W
Esimple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resource:??;simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOp?:simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOp?<simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp?
Dsimple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
6simple_rnn_7/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_7_while_placeholderMsimple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype0?
4simple_rnn_7/while/simple_rnn_cell_7/ones_like/ShapeShape=simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:y
4simple_rnn_7/while/simple_rnn_cell_7/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
.simple_rnn_7/while/simple_rnn_cell_7/ones_likeFill=simple_rnn_7/while/simple_rnn_cell_7/ones_like/Shape:output:0=simple_rnn_7/while/simple_rnn_cell_7/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? ?
6simple_rnn_7/while/simple_rnn_cell_7/ones_like_1/ShapeShape simple_rnn_7_while_placeholder_2*
T0*
_output_shapes
:{
6simple_rnn_7/while/simple_rnn_cell_7/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
0simple_rnn_7/while/simple_rnn_cell_7/ones_like_1Fill?simple_rnn_7/while/simple_rnn_cell_7/ones_like_1/Shape:output:0?simple_rnn_7/while/simple_rnn_cell_7/ones_like_1/Const:output:0*
T0*'
_output_shapes
:??????????
(simple_rnn_7/while/simple_rnn_cell_7/mulMul=simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem:item:07simple_rnn_7/while/simple_rnn_cell_7/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
:simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0?
+simple_rnn_7/while/simple_rnn_cell_7/MatMulMatMul,simple_rnn_7/while/simple_rnn_cell_7/mul:z:0Bsimple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
;simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
,simple_rnn_7/while/simple_rnn_cell_7/BiasAddBiasAdd5simple_rnn_7/while/simple_rnn_cell_7/MatMul:product:0Csimple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
*simple_rnn_7/while/simple_rnn_cell_7/mul_1Mul simple_rnn_7_while_placeholder_29simple_rnn_7/while/simple_rnn_cell_7/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
<simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0?
-simple_rnn_7/while/simple_rnn_cell_7/MatMul_1MatMul.simple_rnn_7/while/simple_rnn_cell_7/mul_1:z:0Dsimple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
(simple_rnn_7/while/simple_rnn_cell_7/addAddV25simple_rnn_7/while/simple_rnn_cell_7/BiasAdd:output:07simple_rnn_7/while/simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:??????????
)simple_rnn_7/while/simple_rnn_cell_7/TanhTanh,simple_rnn_7/while/simple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:??????????
7simple_rnn_7/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_7_while_placeholder_1simple_rnn_7_while_placeholder-simple_rnn_7/while/simple_rnn_cell_7/Tanh:y:0*
_output_shapes
: *
element_dtype0:???Z
simple_rnn_7/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_7/while/addAddV2simple_rnn_7_while_placeholder!simple_rnn_7/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_7/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_7/while/add_1AddV22simple_rnn_7_while_simple_rnn_7_while_loop_counter#simple_rnn_7/while/add_1/y:output:0*
T0*
_output_shapes
: ?
simple_rnn_7/while/IdentityIdentitysimple_rnn_7/while/add_1:z:0^simple_rnn_7/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_7/while/Identity_1Identity8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations^simple_rnn_7/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_7/while/Identity_2Identitysimple_rnn_7/while/add:z:0^simple_rnn_7/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_7/while/Identity_3IdentityGsimple_rnn_7/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_7/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_7/while/Identity_4Identity-simple_rnn_7/while/simple_rnn_cell_7/Tanh:y:0^simple_rnn_7/while/NoOp*
T0*'
_output_shapes
:??????????
simple_rnn_7/while/NoOpNoOp<^simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOp;^simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOp=^simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_7_while_identity$simple_rnn_7/while/Identity:output:0"G
simple_rnn_7_while_identity_1&simple_rnn_7/while/Identity_1:output:0"G
simple_rnn_7_while_identity_2&simple_rnn_7/while/Identity_2:output:0"G
simple_rnn_7_while_identity_3&simple_rnn_7/while/Identity_3:output:0"G
simple_rnn_7_while_identity_4&simple_rnn_7/while/Identity_4:output:0"d
/simple_rnn_7_while_simple_rnn_7_strided_slice_11simple_rnn_7_while_simple_rnn_7_strided_slice_1_0"?
Dsimple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resourceFsimple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resource_0"?
Esimple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resourceGsimple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0"?
Csimple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resourceEsimple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resource_0"?
ksimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensormsimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2z
;simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOp;simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOp2x
:simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOp:simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOp2|
<simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp<simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp: 
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
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
-__inference_simple_rnn_7_layer_call_fn_110026
inputs_0
unknown: 
	unknown_0:
	unknown_1:
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
GPU2*0J 8? *Q
fLRJ
H__inference_simple_rnn_7_layer_call_and_return_conditional_losses_108826o
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
?
H__inference_sequential_7_layer_call_and_return_conditional_losses_109598
embedding_7_input%
embedding_7_109576:	?' %
simple_rnn_7_109579: !
simple_rnn_7_109581:%
simple_rnn_7_109583:!
dense_14_109586:

dense_14_109588:
!
dense_15_109592:

dense_15_109594:
identity?? dense_14/StatefulPartitionedCall? dense_15/StatefulPartitionedCall?!dropout_7/StatefulPartitionedCall?#embedding_7/StatefulPartitionedCall?$simple_rnn_7/StatefulPartitionedCall?
#embedding_7/StatefulPartitionedCallStatefulPartitionedCallembedding_7_inputembedding_7_109576*
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
G__inference_embedding_7_layer_call_and_return_conditional_losses_109035?
$simple_rnn_7/StatefulPartitionedCallStatefulPartitionedCall,embedding_7/StatefulPartitionedCall:output:0simple_rnn_7_109579simple_rnn_7_109581simple_rnn_7_109583*
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
GPU2*0J 8? *Q
fLRJ
H__inference_simple_rnn_7_layer_call_and_return_conditional_losses_109443?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_7/StatefulPartitionedCall:output:0dense_14_109586dense_14_109588*
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
D__inference_dense_14_layer_call_and_return_conditional_losses_109181?
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
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
E__inference_dropout_7_layer_call_and_return_conditional_losses_109261?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0dense_15_109592dense_15_109594*
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
D__inference_dense_15_layer_call_and_return_conditional_losses_109205x
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall$^embedding_7/StatefulPartitionedCall%^simple_rnn_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:??????????????????: : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2J
#embedding_7/StatefulPartitionedCall#embedding_7/StatefulPartitionedCall2L
$simple_rnn_7/StatefulPartitionedCall$simple_rnn_7/StatefulPartitionedCall:c _
0
_output_shapes
:??????????????????
+
_user_specified_nameembedding_7_input
?
?
)__inference_dense_15_layer_call_fn_110675

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
D__inference_dense_15_layer_call_and_return_conditional_losses_109205o
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

?
simple_rnn_7_while_cond_1098856
2simple_rnn_7_while_simple_rnn_7_while_loop_counter<
8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations"
simple_rnn_7_while_placeholder$
 simple_rnn_7_while_placeholder_1$
 simple_rnn_7_while_placeholder_28
4simple_rnn_7_while_less_simple_rnn_7_strided_slice_1N
Jsimple_rnn_7_while_simple_rnn_7_while_cond_109885___redundant_placeholder0N
Jsimple_rnn_7_while_simple_rnn_7_while_cond_109885___redundant_placeholder1N
Jsimple_rnn_7_while_simple_rnn_7_while_cond_109885___redundant_placeholder2N
Jsimple_rnn_7_while_simple_rnn_7_while_cond_109885___redundant_placeholder3
simple_rnn_7_while_identity
?
simple_rnn_7/while/LessLesssimple_rnn_7_while_placeholder4simple_rnn_7_while_less_simple_rnn_7_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_7/while/IdentityIdentitysimple_rnn_7/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_7_while_identity$simple_rnn_7/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????: ::::: 
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
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
M__inference_simple_rnn_cell_7_layer_call_and_return_conditional_losses_108750

inputs

states0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:2
 matmul_1_readvariableop_resource:
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpE
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
:????????? t
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0j
MatMulMatMulmul:z:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????\
mul_1Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:?????????x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0p
MatMul_1MatMul	mul_1:z:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????G
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:????????? :?????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_namestates
?

?
D__inference_dense_15_layer_call_and_return_conditional_losses_109205

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
?Y
?
H__inference_simple_rnn_7_layer_call_and_return_conditional_losses_110339
inputs_0B
0simple_rnn_cell_7_matmul_readvariableop_resource: ?
1simple_rnn_cell_7_biasadd_readvariableop_resource:D
2simple_rnn_cell_7_matmul_1_readvariableop_resource:
identity??(simple_rnn_cell_7/BiasAdd/ReadVariableOp?'simple_rnn_cell_7/MatMul/ReadVariableOp?)simple_rnn_cell_7/MatMul_1/ReadVariableOp?while=
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
shrink_axis_maski
!simple_rnn_cell_7/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:f
!simple_rnn_cell_7/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
simple_rnn_cell_7/ones_likeFill*simple_rnn_cell_7/ones_like/Shape:output:0*simple_rnn_cell_7/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? d
simple_rnn_cell_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
simple_rnn_cell_7/dropout/MulMul$simple_rnn_cell_7/ones_like:output:0(simple_rnn_cell_7/dropout/Const:output:0*
T0*'
_output_shapes
:????????? s
simple_rnn_cell_7/dropout/ShapeShape$simple_rnn_cell_7/ones_like:output:0*
T0*
_output_shapes
:?
6simple_rnn_cell_7/dropout/random_uniform/RandomUniformRandomUniform(simple_rnn_cell_7/dropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2??em
(simple_rnn_cell_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
&simple_rnn_cell_7/dropout/GreaterEqualGreaterEqual?simple_rnn_cell_7/dropout/random_uniform/RandomUniform:output:01simple_rnn_cell_7/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
simple_rnn_cell_7/dropout/CastCast*simple_rnn_cell_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
simple_rnn_cell_7/dropout/Mul_1Mul!simple_rnn_cell_7/dropout/Mul:z:0"simple_rnn_cell_7/dropout/Cast:y:0*
T0*'
_output_shapes
:????????? a
#simple_rnn_cell_7/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:h
#simple_rnn_cell_7/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
simple_rnn_cell_7/ones_like_1Fill,simple_rnn_cell_7/ones_like_1/Shape:output:0,simple_rnn_cell_7/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????f
!simple_rnn_cell_7/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
simple_rnn_cell_7/dropout_1/MulMul&simple_rnn_cell_7/ones_like_1:output:0*simple_rnn_cell_7/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????w
!simple_rnn_cell_7/dropout_1/ShapeShape&simple_rnn_cell_7/ones_like_1:output:0*
T0*
_output_shapes
:?
8simple_rnn_cell_7/dropout_1/random_uniform/RandomUniformRandomUniform*simple_rnn_cell_7/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???o
*simple_rnn_cell_7/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(simple_rnn_cell_7/dropout_1/GreaterEqualGreaterEqualAsimple_rnn_cell_7/dropout_1/random_uniform/RandomUniform:output:03simple_rnn_cell_7/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
 simple_rnn_cell_7/dropout_1/CastCast,simple_rnn_cell_7/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
!simple_rnn_cell_7/dropout_1/Mul_1Mul#simple_rnn_cell_7/dropout_1/Mul:z:0$simple_rnn_cell_7/dropout_1/Cast:y:0*
T0*'
_output_shapes
:??????????
simple_rnn_cell_7/mulMulstrided_slice_2:output:0#simple_rnn_cell_7/dropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
'simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_7_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
simple_rnn_cell_7/MatMulMatMulsimple_rnn_cell_7/mul:z:0/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
(simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
simple_rnn_cell_7/BiasAddBiasAdd"simple_rnn_cell_7/MatMul:product:00simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
simple_rnn_cell_7/mul_1Mulzeros:output:0%simple_rnn_cell_7/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:??????????
)simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_7_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0?
simple_rnn_cell_7/MatMul_1MatMulsimple_rnn_cell_7/mul_1:z:01simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
simple_rnn_cell_7/addAddV2"simple_rnn_cell_7/BiasAdd:output:0$simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:?????????k
simple_rnn_cell_7/TanhTanhsimple_rnn_cell_7/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_7_matmul_readvariableop_resource1simple_rnn_cell_7_biasadd_readvariableop_resource2simple_rnn_cell_7_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_110249*
condR
while_cond_110248*8
output_shapes'
%: : : : :?????????: : : : : *
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
 :??????????????????g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp)^simple_rnn_cell_7/BiasAdd/ReadVariableOp(^simple_rnn_cell_7/MatMul/ReadVariableOp*^simple_rnn_cell_7/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????????????? : : : 2T
(simple_rnn_cell_7/BiasAdd/ReadVariableOp(simple_rnn_cell_7/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_7/MatMul/ReadVariableOp'simple_rnn_cell_7/MatMul/ReadVariableOp2V
)simple_rnn_cell_7/MatMul_1/ReadVariableOp)simple_rnn_cell_7/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :?????????????????? 
"
_user_specified_name
inputs/0
?	
?
G__inference_embedding_7_layer_call_and_return_conditional_losses_109035

inputs*
embedding_lookup_109029:	?' 
identity??embedding_lookup^
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:???????????????????
embedding_lookupResourceGatherembedding_lookup_109029Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/109029*4
_output_shapes"
 :?????????????????? *
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/109029*4
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
?
?
while_cond_110388
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_110388___redundant_placeholder04
0while_while_cond_110388___redundant_placeholder14
0while_while_cond_110388___redundant_placeholder24
0while_while_cond_110388___redundant_placeholder3
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
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????: ::::: 
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
:?????????:

_output_shapes
: :

_output_shapes
:
?

?
-__inference_sequential_7_layer_call_fn_109548
embedding_7_input
unknown:	?' 
	unknown_0: 
	unknown_1:
	unknown_2:
	unknown_3:

	unknown_4:

	unknown_5:

	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallembedding_7_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
H__inference_sequential_7_layer_call_and_return_conditional_losses_109508o
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
_user_specified_nameembedding_7_input
?
?
while_cond_109352
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_109352___redundant_placeholder04
0while_while_cond_109352___redundant_placeholder14
0while_while_cond_109352___redundant_placeholder24
0while_while_cond_109352___redundant_placeholder3
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
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????: ::::: 
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
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_110528
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_110528___redundant_placeholder04
0while_while_cond_110528___redundant_placeholder14
0while_while_cond_110528___redundant_placeholder24
0while_while_cond_110528___redundant_placeholder3
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
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????: ::::: 
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
:?????????:

_output_shapes
: :

_output_shapes
:
?J
?
while_body_110249
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_7_matmul_readvariableop_resource_0: G
9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0:L
:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_7_matmul_readvariableop_resource: E
7while_simple_rnn_cell_7_biasadd_readvariableop_resource:J
8while_simple_rnn_cell_7_matmul_1_readvariableop_resource:??.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_7/MatMul/ReadVariableOp?/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype0?
'while/simple_rnn_cell_7/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:l
'while/simple_rnn_cell_7/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
!while/simple_rnn_cell_7/ones_likeFill0while/simple_rnn_cell_7/ones_like/Shape:output:00while/simple_rnn_cell_7/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? j
%while/simple_rnn_cell_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#while/simple_rnn_cell_7/dropout/MulMul*while/simple_rnn_cell_7/ones_like:output:0.while/simple_rnn_cell_7/dropout/Const:output:0*
T0*'
_output_shapes
:????????? 
%while/simple_rnn_cell_7/dropout/ShapeShape*while/simple_rnn_cell_7/ones_like:output:0*
T0*
_output_shapes
:?
<while/simple_rnn_cell_7/dropout/random_uniform/RandomUniformRandomUniform.while/simple_rnn_cell_7/dropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2???s
.while/simple_rnn_cell_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
,while/simple_rnn_cell_7/dropout/GreaterEqualGreaterEqualEwhile/simple_rnn_cell_7/dropout/random_uniform/RandomUniform:output:07while/simple_rnn_cell_7/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
$while/simple_rnn_cell_7/dropout/CastCast0while/simple_rnn_cell_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
%while/simple_rnn_cell_7/dropout/Mul_1Mul'while/simple_rnn_cell_7/dropout/Mul:z:0(while/simple_rnn_cell_7/dropout/Cast:y:0*
T0*'
_output_shapes
:????????? l
)while/simple_rnn_cell_7/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:n
)while/simple_rnn_cell_7/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#while/simple_rnn_cell_7/ones_like_1Fill2while/simple_rnn_cell_7/ones_like_1/Shape:output:02while/simple_rnn_cell_7/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????l
'while/simple_rnn_cell_7/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
%while/simple_rnn_cell_7/dropout_1/MulMul,while/simple_rnn_cell_7/ones_like_1:output:00while/simple_rnn_cell_7/dropout_1/Const:output:0*
T0*'
_output_shapes
:??????????
'while/simple_rnn_cell_7/dropout_1/ShapeShape,while/simple_rnn_cell_7/ones_like_1:output:0*
T0*
_output_shapes
:?
>while/simple_rnn_cell_7/dropout_1/random_uniform/RandomUniformRandomUniform0while/simple_rnn_cell_7/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2º?u
0while/simple_rnn_cell_7/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
.while/simple_rnn_cell_7/dropout_1/GreaterEqualGreaterEqualGwhile/simple_rnn_cell_7/dropout_1/random_uniform/RandomUniform:output:09while/simple_rnn_cell_7/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
&while/simple_rnn_cell_7/dropout_1/CastCast2while/simple_rnn_cell_7/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
'while/simple_rnn_cell_7/dropout_1/Mul_1Mul)while/simple_rnn_cell_7/dropout_1/Mul:z:0*while/simple_rnn_cell_7/dropout_1/Cast:y:0*
T0*'
_output_shapes
:??????????
while/simple_rnn_cell_7/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/simple_rnn_cell_7/dropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
-while/simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_7_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0?
while/simple_rnn_cell_7/MatMulMatMulwhile/simple_rnn_cell_7/mul:z:05while/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
.while/simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/simple_rnn_cell_7/BiasAddBiasAdd(while/simple_rnn_cell_7/MatMul:product:06while/simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
while/simple_rnn_cell_7/mul_1Mulwhile_placeholder_2+while/simple_rnn_cell_7/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:??????????
/while/simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0?
 while/simple_rnn_cell_7/MatMul_1MatMul!while/simple_rnn_cell_7/mul_1:z:07while/simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
while/simple_rnn_cell_7/addAddV2(while/simple_rnn_cell_7/BiasAdd:output:0*while/simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:?????????w
while/simple_rnn_cell_7/TanhTanhwhile/simple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_7/Tanh:y:0*
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
: }
while/Identity_4Identity while/simple_rnn_cell_7/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp/^while/simple_rnn_cell_7/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_7/MatMul/ReadVariableOp0^while/simple_rnn_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_7_biasadd_readvariableop_resource9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_7_matmul_1_readvariableop_resource:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_7_matmul_readvariableop_resource8while_simple_rnn_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2`
.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_7/MatMul/ReadVariableOp-while/simple_rnn_cell_7/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp: 
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
:?????????:

_output_shapes
: :

_output_shapes
: 
?Y
?
H__inference_simple_rnn_7_layer_call_and_return_conditional_losses_109443

inputsB
0simple_rnn_cell_7_matmul_readvariableop_resource: ?
1simple_rnn_cell_7_biasadd_readvariableop_resource:D
2simple_rnn_cell_7_matmul_1_readvariableop_resource:
identity??(simple_rnn_cell_7/BiasAdd/ReadVariableOp?'simple_rnn_cell_7/MatMul/ReadVariableOp?)simple_rnn_cell_7/MatMul_1/ReadVariableOp?while;
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
shrink_axis_maski
!simple_rnn_cell_7/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:f
!simple_rnn_cell_7/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
simple_rnn_cell_7/ones_likeFill*simple_rnn_cell_7/ones_like/Shape:output:0*simple_rnn_cell_7/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? d
simple_rnn_cell_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
simple_rnn_cell_7/dropout/MulMul$simple_rnn_cell_7/ones_like:output:0(simple_rnn_cell_7/dropout/Const:output:0*
T0*'
_output_shapes
:????????? s
simple_rnn_cell_7/dropout/ShapeShape$simple_rnn_cell_7/ones_like:output:0*
T0*
_output_shapes
:?
6simple_rnn_cell_7/dropout/random_uniform/RandomUniformRandomUniform(simple_rnn_cell_7/dropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2̋?m
(simple_rnn_cell_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
&simple_rnn_cell_7/dropout/GreaterEqualGreaterEqual?simple_rnn_cell_7/dropout/random_uniform/RandomUniform:output:01simple_rnn_cell_7/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
simple_rnn_cell_7/dropout/CastCast*simple_rnn_cell_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
simple_rnn_cell_7/dropout/Mul_1Mul!simple_rnn_cell_7/dropout/Mul:z:0"simple_rnn_cell_7/dropout/Cast:y:0*
T0*'
_output_shapes
:????????? a
#simple_rnn_cell_7/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:h
#simple_rnn_cell_7/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
simple_rnn_cell_7/ones_like_1Fill,simple_rnn_cell_7/ones_like_1/Shape:output:0,simple_rnn_cell_7/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????f
!simple_rnn_cell_7/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
simple_rnn_cell_7/dropout_1/MulMul&simple_rnn_cell_7/ones_like_1:output:0*simple_rnn_cell_7/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????w
!simple_rnn_cell_7/dropout_1/ShapeShape&simple_rnn_cell_7/ones_like_1:output:0*
T0*
_output_shapes
:?
8simple_rnn_cell_7/dropout_1/random_uniform/RandomUniformRandomUniform*simple_rnn_cell_7/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2??Do
*simple_rnn_cell_7/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(simple_rnn_cell_7/dropout_1/GreaterEqualGreaterEqualAsimple_rnn_cell_7/dropout_1/random_uniform/RandomUniform:output:03simple_rnn_cell_7/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
 simple_rnn_cell_7/dropout_1/CastCast,simple_rnn_cell_7/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
!simple_rnn_cell_7/dropout_1/Mul_1Mul#simple_rnn_cell_7/dropout_1/Mul:z:0$simple_rnn_cell_7/dropout_1/Cast:y:0*
T0*'
_output_shapes
:??????????
simple_rnn_cell_7/mulMulstrided_slice_2:output:0#simple_rnn_cell_7/dropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
'simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_7_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
simple_rnn_cell_7/MatMulMatMulsimple_rnn_cell_7/mul:z:0/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
(simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
simple_rnn_cell_7/BiasAddBiasAdd"simple_rnn_cell_7/MatMul:product:00simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
simple_rnn_cell_7/mul_1Mulzeros:output:0%simple_rnn_cell_7/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:??????????
)simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_7_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0?
simple_rnn_cell_7/MatMul_1MatMulsimple_rnn_cell_7/mul_1:z:01simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
simple_rnn_cell_7/addAddV2"simple_rnn_cell_7/BiasAdd:output:0$simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:?????????k
simple_rnn_cell_7/TanhTanhsimple_rnn_cell_7/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_7_matmul_readvariableop_resource1simple_rnn_cell_7_biasadd_readvariableop_resource2simple_rnn_cell_7_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_109353*
condR
while_cond_109352*8
output_shapes'
%: : : : :?????????: : : : : *
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
 :??????????????????g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp)^simple_rnn_cell_7/BiasAdd/ReadVariableOp(^simple_rnn_cell_7/MatMul/ReadVariableOp*^simple_rnn_cell_7/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????????????? : : : 2T
(simple_rnn_cell_7/BiasAdd/ReadVariableOp(simple_rnn_cell_7/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_7/MatMul/ReadVariableOp'simple_rnn_cell_7/MatMul/ReadVariableOp2V
)simple_rnn_cell_7/MatMul_1/ReadVariableOp)simple_rnn_cell_7/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?

?
-__inference_sequential_7_layer_call_fn_109231
embedding_7_input
unknown:	?' 
	unknown_0: 
	unknown_1:
	unknown_2:
	unknown_3:

	unknown_4:

	unknown_5:

	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallembedding_7_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
H__inference_sequential_7_layer_call_and_return_conditional_losses_109212o
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
_user_specified_nameembedding_7_input
?
c
*__inference_dropout_7_layer_call_fn_110649

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
E__inference_dropout_7_layer_call_and_return_conditional_losses_109261o
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
?J
?
while_body_110529
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_7_matmul_readvariableop_resource_0: G
9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0:L
:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_7_matmul_readvariableop_resource: E
7while_simple_rnn_cell_7_biasadd_readvariableop_resource:J
8while_simple_rnn_cell_7_matmul_1_readvariableop_resource:??.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp?-while/simple_rnn_cell_7/MatMul/ReadVariableOp?/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype0?
'while/simple_rnn_cell_7/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:l
'while/simple_rnn_cell_7/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
!while/simple_rnn_cell_7/ones_likeFill0while/simple_rnn_cell_7/ones_like/Shape:output:00while/simple_rnn_cell_7/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? j
%while/simple_rnn_cell_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#while/simple_rnn_cell_7/dropout/MulMul*while/simple_rnn_cell_7/ones_like:output:0.while/simple_rnn_cell_7/dropout/Const:output:0*
T0*'
_output_shapes
:????????? 
%while/simple_rnn_cell_7/dropout/ShapeShape*while/simple_rnn_cell_7/ones_like:output:0*
T0*
_output_shapes
:?
<while/simple_rnn_cell_7/dropout/random_uniform/RandomUniformRandomUniform.while/simple_rnn_cell_7/dropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*
seed???)*
seed2??=s
.while/simple_rnn_cell_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
,while/simple_rnn_cell_7/dropout/GreaterEqualGreaterEqualEwhile/simple_rnn_cell_7/dropout/random_uniform/RandomUniform:output:07while/simple_rnn_cell_7/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? ?
$while/simple_rnn_cell_7/dropout/CastCast0while/simple_rnn_cell_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
%while/simple_rnn_cell_7/dropout/Mul_1Mul'while/simple_rnn_cell_7/dropout/Mul:z:0(while/simple_rnn_cell_7/dropout/Cast:y:0*
T0*'
_output_shapes
:????????? l
)while/simple_rnn_cell_7/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:n
)while/simple_rnn_cell_7/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#while/simple_rnn_cell_7/ones_like_1Fill2while/simple_rnn_cell_7/ones_like_1/Shape:output:02while/simple_rnn_cell_7/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????l
'while/simple_rnn_cell_7/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
%while/simple_rnn_cell_7/dropout_1/MulMul,while/simple_rnn_cell_7/ones_like_1:output:00while/simple_rnn_cell_7/dropout_1/Const:output:0*
T0*'
_output_shapes
:??????????
'while/simple_rnn_cell_7/dropout_1/ShapeShape,while/simple_rnn_cell_7/ones_like_1:output:0*
T0*
_output_shapes
:?
>while/simple_rnn_cell_7/dropout_1/random_uniform/RandomUniformRandomUniform0while/simple_rnn_cell_7/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???u
0while/simple_rnn_cell_7/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
.while/simple_rnn_cell_7/dropout_1/GreaterEqualGreaterEqualGwhile/simple_rnn_cell_7/dropout_1/random_uniform/RandomUniform:output:09while/simple_rnn_cell_7/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
&while/simple_rnn_cell_7/dropout_1/CastCast2while/simple_rnn_cell_7/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
'while/simple_rnn_cell_7/dropout_1/Mul_1Mul)while/simple_rnn_cell_7/dropout_1/Mul:z:0*while/simple_rnn_cell_7/dropout_1/Cast:y:0*
T0*'
_output_shapes
:??????????
while/simple_rnn_cell_7/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/simple_rnn_cell_7/dropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? ?
-while/simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_7_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0?
while/simple_rnn_cell_7/MatMulMatMulwhile/simple_rnn_cell_7/mul:z:05while/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
.while/simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
while/simple_rnn_cell_7/BiasAddBiasAdd(while/simple_rnn_cell_7/MatMul:product:06while/simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
while/simple_rnn_cell_7/mul_1Mulwhile_placeholder_2+while/simple_rnn_cell_7/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:??????????
/while/simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0?
 while/simple_rnn_cell_7/MatMul_1MatMul!while/simple_rnn_cell_7/mul_1:z:07while/simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
while/simple_rnn_cell_7/addAddV2(while/simple_rnn_cell_7/BiasAdd:output:0*while/simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:?????????w
while/simple_rnn_cell_7/TanhTanhwhile/simple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:??????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_7/Tanh:y:0*
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
: }
while/Identity_4Identity while/simple_rnn_cell_7/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:??????????

while/NoOpNoOp/^while/simple_rnn_cell_7/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_7/MatMul/ReadVariableOp0^while/simple_rnn_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_7_biasadd_readvariableop_resource9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_7_matmul_1_readvariableop_resource:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_7_matmul_readvariableop_resource8while_simple_rnn_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2`
.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_7/MatMul/ReadVariableOp-while/simple_rnn_cell_7/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp: 
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
:?????????:

_output_shapes
: :

_output_shapes
: 
?Q
?
+sequential_7_simple_rnn_7_while_body_108605P
Lsequential_7_simple_rnn_7_while_sequential_7_simple_rnn_7_while_loop_counterV
Rsequential_7_simple_rnn_7_while_sequential_7_simple_rnn_7_while_maximum_iterations/
+sequential_7_simple_rnn_7_while_placeholder1
-sequential_7_simple_rnn_7_while_placeholder_11
-sequential_7_simple_rnn_7_while_placeholder_2O
Ksequential_7_simple_rnn_7_while_sequential_7_simple_rnn_7_strided_slice_1_0?
?sequential_7_simple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_sequential_7_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0d
Rsequential_7_simple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resource_0: a
Ssequential_7_simple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resource_0:f
Tsequential_7_simple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0:,
(sequential_7_simple_rnn_7_while_identity.
*sequential_7_simple_rnn_7_while_identity_1.
*sequential_7_simple_rnn_7_while_identity_2.
*sequential_7_simple_rnn_7_while_identity_3.
*sequential_7_simple_rnn_7_while_identity_4M
Isequential_7_simple_rnn_7_while_sequential_7_simple_rnn_7_strided_slice_1?
?sequential_7_simple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_sequential_7_simple_rnn_7_tensorarrayunstack_tensorlistfromtensorb
Psequential_7_simple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resource: _
Qsequential_7_simple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resource:d
Rsequential_7_simple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resource:??Hsequential_7/simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOp?Gsequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOp?Isequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp?
Qsequential_7/simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    ?
Csequential_7/simple_rnn_7/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?sequential_7_simple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_sequential_7_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0+sequential_7_simple_rnn_7_while_placeholderZsequential_7/simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype0?
Asequential_7/simple_rnn_7/while/simple_rnn_cell_7/ones_like/ShapeShapeJsequential_7/simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:?
Asequential_7/simple_rnn_7/while/simple_rnn_cell_7/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
;sequential_7/simple_rnn_7/while/simple_rnn_cell_7/ones_likeFillJsequential_7/simple_rnn_7/while/simple_rnn_cell_7/ones_like/Shape:output:0Jsequential_7/simple_rnn_7/while/simple_rnn_cell_7/ones_like/Const:output:0*
T0*'
_output_shapes
:????????? ?
Csequential_7/simple_rnn_7/while/simple_rnn_cell_7/ones_like_1/ShapeShape-sequential_7_simple_rnn_7_while_placeholder_2*
T0*
_output_shapes
:?
Csequential_7/simple_rnn_7/while/simple_rnn_cell_7/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
=sequential_7/simple_rnn_7/while/simple_rnn_cell_7/ones_like_1FillLsequential_7/simple_rnn_7/while/simple_rnn_cell_7/ones_like_1/Shape:output:0Lsequential_7/simple_rnn_7/while/simple_rnn_cell_7/ones_like_1/Const:output:0*
T0*'
_output_shapes
:??????????
5sequential_7/simple_rnn_7/while/simple_rnn_cell_7/mulMulJsequential_7/simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential_7/simple_rnn_7/while/simple_rnn_cell_7/ones_like:output:0*
T0*'
_output_shapes
:????????? ?
Gsequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOpRsequential_7_simple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0?
8sequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMulMatMul9sequential_7/simple_rnn_7/while/simple_rnn_cell_7/mul:z:0Osequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Hsequential_7/simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOpSsequential_7_simple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0?
9sequential_7/simple_rnn_7/while/simple_rnn_cell_7/BiasAddBiasAddBsequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul:product:0Psequential_7/simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
7sequential_7/simple_rnn_7/while/simple_rnn_cell_7/mul_1Mul-sequential_7_simple_rnn_7_while_placeholder_2Fsequential_7/simple_rnn_7/while/simple_rnn_cell_7/ones_like_1:output:0*
T0*'
_output_shapes
:??????????
Isequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOpTsequential_7_simple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0?
:sequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul_1MatMul;sequential_7/simple_rnn_7/while/simple_rnn_cell_7/mul_1:z:0Qsequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
5sequential_7/simple_rnn_7/while/simple_rnn_cell_7/addAddV2Bsequential_7/simple_rnn_7/while/simple_rnn_cell_7/BiasAdd:output:0Dsequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:??????????
6sequential_7/simple_rnn_7/while/simple_rnn_cell_7/TanhTanh9sequential_7/simple_rnn_7/while/simple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:??????????
Dsequential_7/simple_rnn_7/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem-sequential_7_simple_rnn_7_while_placeholder_1+sequential_7_simple_rnn_7_while_placeholder:sequential_7/simple_rnn_7/while/simple_rnn_cell_7/Tanh:y:0*
_output_shapes
: *
element_dtype0:???g
%sequential_7/simple_rnn_7/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
#sequential_7/simple_rnn_7/while/addAddV2+sequential_7_simple_rnn_7_while_placeholder.sequential_7/simple_rnn_7/while/add/y:output:0*
T0*
_output_shapes
: i
'sequential_7/simple_rnn_7/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
%sequential_7/simple_rnn_7/while/add_1AddV2Lsequential_7_simple_rnn_7_while_sequential_7_simple_rnn_7_while_loop_counter0sequential_7/simple_rnn_7/while/add_1/y:output:0*
T0*
_output_shapes
: ?
(sequential_7/simple_rnn_7/while/IdentityIdentity)sequential_7/simple_rnn_7/while/add_1:z:0%^sequential_7/simple_rnn_7/while/NoOp*
T0*
_output_shapes
: ?
*sequential_7/simple_rnn_7/while/Identity_1IdentityRsequential_7_simple_rnn_7_while_sequential_7_simple_rnn_7_while_maximum_iterations%^sequential_7/simple_rnn_7/while/NoOp*
T0*
_output_shapes
: ?
*sequential_7/simple_rnn_7/while/Identity_2Identity'sequential_7/simple_rnn_7/while/add:z:0%^sequential_7/simple_rnn_7/while/NoOp*
T0*
_output_shapes
: ?
*sequential_7/simple_rnn_7/while/Identity_3IdentityTsequential_7/simple_rnn_7/while/TensorArrayV2Write/TensorListSetItem:output_handle:0%^sequential_7/simple_rnn_7/while/NoOp*
T0*
_output_shapes
: ?
*sequential_7/simple_rnn_7/while/Identity_4Identity:sequential_7/simple_rnn_7/while/simple_rnn_cell_7/Tanh:y:0%^sequential_7/simple_rnn_7/while/NoOp*
T0*'
_output_shapes
:??????????
$sequential_7/simple_rnn_7/while/NoOpNoOpI^sequential_7/simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOpH^sequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOpJ^sequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "]
(sequential_7_simple_rnn_7_while_identity1sequential_7/simple_rnn_7/while/Identity:output:0"a
*sequential_7_simple_rnn_7_while_identity_13sequential_7/simple_rnn_7/while/Identity_1:output:0"a
*sequential_7_simple_rnn_7_while_identity_23sequential_7/simple_rnn_7/while/Identity_2:output:0"a
*sequential_7_simple_rnn_7_while_identity_33sequential_7/simple_rnn_7/while/Identity_3:output:0"a
*sequential_7_simple_rnn_7_while_identity_43sequential_7/simple_rnn_7/while/Identity_4:output:0"?
Isequential_7_simple_rnn_7_while_sequential_7_simple_rnn_7_strided_slice_1Ksequential_7_simple_rnn_7_while_sequential_7_simple_rnn_7_strided_slice_1_0"?
Qsequential_7_simple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resourceSsequential_7_simple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resource_0"?
Rsequential_7_simple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resourceTsequential_7_simple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0"?
Psequential_7_simple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resourceRsequential_7_simple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resource_0"?
?sequential_7_simple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_sequential_7_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor?sequential_7_simple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_sequential_7_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????: : : : : 2?
Hsequential_7/simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOpHsequential_7/simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOp2?
Gsequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOpGsequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOp2?
Isequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOpIsequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp: 
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
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
+sequential_7_simple_rnn_7_while_cond_108604P
Lsequential_7_simple_rnn_7_while_sequential_7_simple_rnn_7_while_loop_counterV
Rsequential_7_simple_rnn_7_while_sequential_7_simple_rnn_7_while_maximum_iterations/
+sequential_7_simple_rnn_7_while_placeholder1
-sequential_7_simple_rnn_7_while_placeholder_11
-sequential_7_simple_rnn_7_while_placeholder_2R
Nsequential_7_simple_rnn_7_while_less_sequential_7_simple_rnn_7_strided_slice_1h
dsequential_7_simple_rnn_7_while_sequential_7_simple_rnn_7_while_cond_108604___redundant_placeholder0h
dsequential_7_simple_rnn_7_while_sequential_7_simple_rnn_7_while_cond_108604___redundant_placeholder1h
dsequential_7_simple_rnn_7_while_sequential_7_simple_rnn_7_while_cond_108604___redundant_placeholder2h
dsequential_7_simple_rnn_7_while_sequential_7_simple_rnn_7_while_cond_108604___redundant_placeholder3,
(sequential_7_simple_rnn_7_while_identity
?
$sequential_7/simple_rnn_7/while/LessLess+sequential_7_simple_rnn_7_while_placeholderNsequential_7_simple_rnn_7_while_less_sequential_7_simple_rnn_7_strided_slice_1*
T0*
_output_shapes
: 
(sequential_7/simple_rnn_7/while/IdentityIdentity(sequential_7/simple_rnn_7/while/Less:z:0*
T0
*
_output_shapes
: "]
(sequential_7_simple_rnn_7_while_identity1sequential_7/simple_rnn_7/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????: ::::: 
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
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
-__inference_simple_rnn_7_layer_call_fn_110048

inputs
unknown: 
	unknown_0:
	unknown_1:
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
GPU2*0J 8? *Q
fLRJ
H__inference_simple_rnn_7_layer_call_and_return_conditional_losses_109162o
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
?

?
D__inference_dense_14_layer_call_and_return_conditional_losses_110639

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
D__inference_dense_15_layer_call_and_return_conditional_losses_110686

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

 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
X
embedding_7_inputC
#serving_default_embedding_7_input:0??????????????????<
dense_150
StatefulPartitionedCall:0?????????tensorflow/serving/predict:Ց
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
r__call__
*s&call_and_return_all_conditional_losses
t_default_save_signature"
_tf_keras_sequential
?

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
u__call__
*v&call_and_return_all_conditional_losses"
_tf_keras_layer
?
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
w__call__
*x&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
y__call__
*z&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
 	keras_api
{__call__
*|&call_and_return_all_conditional_losses"
_tf_keras_layer
?

!kernel
"bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
}__call__
*~&call_and_return_all_conditional_losses"
_tf_keras_layer
?
'iter

(beta_1

)beta_2
	*decay
+learning_ratembmcmd!me"mf,mg-mh.mivjvkvl!vm"vn,vo-vp.vq"
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
r__call__
t_default_save_signature
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
,
serving_default"
signature_map
):'	?' 2embedding_7/embeddings
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
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
?

,kernel
-recurrent_kernel
.bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
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

=states
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
!:
2dense_14/kernel
:
2dense_14/bias
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
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
!:
2dense_15/kernel
:2dense_15/bias
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
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
#	variables
$trainable_variables
%regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
7:5 2%simple_rnn_7/simple_rnn_cell_7/kernel
A:?2/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel
1:/2#simple_rnn_7/simple_rnn_cell_7/bias
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
R0
S1"
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
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
9	variables
:trainable_variables
;regularization_losses
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
	Ytotal
	Zcount
[	variables
\	keras_api"
_tf_keras_metric
^
	]total
	^count
_
_fn_kwargs
`	variables
a	keras_api"
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
Y0
Z1"
trackable_list_wrapper
-
[	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
]0
^1"
trackable_list_wrapper
-
`	variables"
_generic_user_object
.:,	?' 2Adam/embedding_7/embeddings/m
&:$
2Adam/dense_14/kernel/m
 :
2Adam/dense_14/bias/m
&:$
2Adam/dense_15/kernel/m
 :2Adam/dense_15/bias/m
<:: 2,Adam/simple_rnn_7/simple_rnn_cell_7/kernel/m
F:D26Adam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/m
6:42*Adam/simple_rnn_7/simple_rnn_cell_7/bias/m
.:,	?' 2Adam/embedding_7/embeddings/v
&:$
2Adam/dense_14/kernel/v
 :
2Adam/dense_14/bias/v
&:$
2Adam/dense_15/kernel/v
 :2Adam/dense_15/bias/v
<:: 2,Adam/simple_rnn_7/simple_rnn_cell_7/kernel/v
F:D26Adam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/v
6:42*Adam/simple_rnn_7/simple_rnn_cell_7/bias/v
?2?
-__inference_sequential_7_layer_call_fn_109231
-__inference_sequential_7_layer_call_fn_109648
-__inference_sequential_7_layer_call_fn_109669
-__inference_sequential_7_layer_call_fn_109548?
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
H__inference_sequential_7_layer_call_and_return_conditional_losses_109814
H__inference_sequential_7_layer_call_and_return_conditional_losses_109998
H__inference_sequential_7_layer_call_and_return_conditional_losses_109573
H__inference_sequential_7_layer_call_and_return_conditional_losses_109598?
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
!__inference__wrapped_model_108694embedding_7_input"?
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
,__inference_embedding_7_layer_call_fn_110005?
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
G__inference_embedding_7_layer_call_and_return_conditional_losses_110015?
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
?2?
-__inference_simple_rnn_7_layer_call_fn_110026
-__inference_simple_rnn_7_layer_call_fn_110037
-__inference_simple_rnn_7_layer_call_fn_110048
-__inference_simple_rnn_7_layer_call_fn_110059?
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
?2?
H__inference_simple_rnn_7_layer_call_and_return_conditional_losses_110183
H__inference_simple_rnn_7_layer_call_and_return_conditional_losses_110339
H__inference_simple_rnn_7_layer_call_and_return_conditional_losses_110463
H__inference_simple_rnn_7_layer_call_and_return_conditional_losses_110619?
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
)__inference_dense_14_layer_call_fn_110628?
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
D__inference_dense_14_layer_call_and_return_conditional_losses_110639?
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
*__inference_dropout_7_layer_call_fn_110644
*__inference_dropout_7_layer_call_fn_110649?
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
E__inference_dropout_7_layer_call_and_return_conditional_losses_110654
E__inference_dropout_7_layer_call_and_return_conditional_losses_110666?
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
)__inference_dense_15_layer_call_fn_110675?
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
D__inference_dense_15_layer_call_and_return_conditional_losses_110686?
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
$__inference_signature_wrapper_109627embedding_7_input"?
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
2__inference_simple_rnn_cell_7_layer_call_fn_110700
2__inference_simple_rnn_cell_7_layer_call_fn_110714?
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
M__inference_simple_rnn_cell_7_layer_call_and_return_conditional_losses_110739
M__inference_simple_rnn_cell_7_layer_call_and_return_conditional_losses_110780?
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
!__inference__wrapped_model_108694?,.-!"C?@
9?6
4?1
embedding_7_input??????????????????
? "3?0
.
dense_15"?
dense_15??????????
D__inference_dense_14_layer_call_and_return_conditional_losses_110639\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????

? |
)__inference_dense_14_layer_call_fn_110628O/?,
%?"
 ?
inputs?????????
? "??????????
?
D__inference_dense_15_layer_call_and_return_conditional_losses_110686\!"/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????
? |
)__inference_dense_15_layer_call_fn_110675O!"/?,
%?"
 ?
inputs?????????

? "???????????
E__inference_dropout_7_layer_call_and_return_conditional_losses_110654\3?0
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
E__inference_dropout_7_layer_call_and_return_conditional_losses_110666\3?0
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
*__inference_dropout_7_layer_call_fn_110644O3?0
)?&
 ?
inputs?????????

p 
? "??????????
}
*__inference_dropout_7_layer_call_fn_110649O3?0
)?&
 ?
inputs?????????

p
? "??????????
?
G__inference_embedding_7_layer_call_and_return_conditional_losses_110015q8?5
.?+
)?&
inputs??????????????????
? "2?/
(?%
0?????????????????? 
? ?
,__inference_embedding_7_layer_call_fn_110005d8?5
.?+
)?&
inputs??????????????????
? "%?"?????????????????? ?
H__inference_sequential_7_layer_call_and_return_conditional_losses_109573~,.-!"K?H
A?>
4?1
embedding_7_input??????????????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_7_layer_call_and_return_conditional_losses_109598~,.-!"K?H
A?>
4?1
embedding_7_input??????????????????
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_7_layer_call_and_return_conditional_losses_109814s,.-!"@?=
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
H__inference_sequential_7_layer_call_and_return_conditional_losses_109998s,.-!"@?=
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
-__inference_sequential_7_layer_call_fn_109231q,.-!"K?H
A?>
4?1
embedding_7_input??????????????????
p 

 
? "???????????
-__inference_sequential_7_layer_call_fn_109548q,.-!"K?H
A?>
4?1
embedding_7_input??????????????????
p

 
? "???????????
-__inference_sequential_7_layer_call_fn_109648f,.-!"@?=
6?3
)?&
inputs??????????????????
p 

 
? "???????????
-__inference_sequential_7_layer_call_fn_109669f,.-!"@?=
6?3
)?&
inputs??????????????????
p

 
? "???????????
$__inference_signature_wrapper_109627?,.-!"X?U
? 
N?K
I
embedding_7_input4?1
embedding_7_input??????????????????"3?0
.
dense_15"?
dense_15??????????
H__inference_simple_rnn_7_layer_call_and_return_conditional_losses_110183},.-O?L
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
H__inference_simple_rnn_7_layer_call_and_return_conditional_losses_110339},.-O?L
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
H__inference_simple_rnn_7_layer_call_and_return_conditional_losses_110463v,.-H?E
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
H__inference_simple_rnn_7_layer_call_and_return_conditional_losses_110619v,.-H?E
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
-__inference_simple_rnn_7_layer_call_fn_110026p,.-O?L
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
-__inference_simple_rnn_7_layer_call_fn_110037p,.-O?L
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
-__inference_simple_rnn_7_layer_call_fn_110048i,.-H?E
>?;
-?*
inputs?????????????????? 

 
p 

 
? "???????????
-__inference_simple_rnn_7_layer_call_fn_110059i,.-H?E
>?;
-?*
inputs?????????????????? 

 
p

 
? "???????????
M__inference_simple_rnn_cell_7_layer_call_and_return_conditional_losses_110739?,.-\?Y
R?O
 ?
inputs????????? 
'?$
"?
states/0?????????
p 
? "R?O
H?E
?
0/0?????????
$?!
?
0/1/0?????????
? ?
M__inference_simple_rnn_cell_7_layer_call_and_return_conditional_losses_110780?,.-\?Y
R?O
 ?
inputs????????? 
'?$
"?
states/0?????????
p
? "R?O
H?E
?
0/0?????????
$?!
?
0/1/0?????????
? ?
2__inference_simple_rnn_cell_7_layer_call_fn_110700?,.-\?Y
R?O
 ?
inputs????????? 
'?$
"?
states/0?????????
p 
? "D?A
?
0?????????
"?
?
1/0??????????
2__inference_simple_rnn_cell_7_layer_call_fn_110714?,.-\?Y
R?O
 ?
inputs????????? 
'?$
"?
states/0?????????
p
? "D?A
?
0?????????
"?
?
1/0?????????