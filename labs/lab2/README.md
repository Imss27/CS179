CS 179: GPU Computing
Lab 2: Matrix transpose
Name: Imss27

==============================================================================================
Question 1.1: Latency Hiding (5 points)
==============================================================================================

--------------------------------------------------------------------------------------------------
1.1
Answer : According to the slides, the latency of an arithmetic instruction(warp add) is 10 cycles, and a GK110 can start 8 instructions per cycle(2 ILP and 4 warps scheduler). So to hide this latency, 80 instructions are needed.
--------------------------------------------------------------------------------------------------

==============================================================================================
Question 1.2: Thread Divergence (6 points)
==============================================================================================

--------------------------------------------------------------------------------------------------
1.2
(a) 
Answer : No, there is no warp divergence.
Because int idx = threadIdx.y + 32 * threadIdx.x; So idx % 32 is decided by threadIdx.y only.
And this makes all threads in a warp execute in the same branch.

(b)
Answer : Yes, the code diverges: From a strict perspective, threads execute different numbers of iterations based on threadIdx.x, leading to different execution paths within the same warp.
--------------------------------------------------------------------------------------------------

==============================================================================================
Question 1.3: Coalesced Memory Access (9 points)
==============================================================================================

--------------------------------------------------------------------------------------------------
1.3
(a)
Answer : It is coalesced. The threads in the warp write in the same cache line, it writes
to only one cache line.

(b)
Answer : It is not coalesced. The threads in the warp access the data with step 32 * 4 = 128 bytes, 
so it writes to 32 cache lines.

(c)
Answer : It write is not coalesced. It writes to two cache lines.
--------------------------------------------------------------------------------------------------


==============================================================================================
Question 1.4: Bank Conflicts and Instruction Dependencies (15 points)
==============================================================================================


--------------------------------------------------------------------------------------------------
1.4
(a) 
Answer : 
    There are no bank conflicts in this code. Since thread(i0, j) and thread(i1, j) are accessing different banks for lhs and the same banks for rhs which can utilize broadcast.
    ??? not sure about this answer.

(b)
Answer :
    lhs0 = lhs[i + 32 * k];
    rhs0 = rhs[k + 128 * j];
    O0 = output[i + 32 * j];
    FMA on lhs0, rhs0, O0;
    Write O0 to output[i + 32 * j];

    lhs1 = lhs[i + 32 * (k + 1)];
    rhs1 = rhs[(k + 1) + 128 * j];
    O1 = output[i + 32 * j];
    FMA on lhs1, rhs1, O1;
    Write O1 to output[i + 32 * j];


(c)
Answer :
“Write O0 to output[i + 32 * j];” depends on “FMA on lhs0, rhs0, O0;”. “Write O1 to output[i + 32 * j];” depends on “FMA on lhs1, rhs1, O1;”. “FMA on lhs0, rhs0, O0;” depends on “lhs0, rhs0, and O0”. “FMA on lhs1, rhs1, O1;” depends on “lhs1, rhs1, and O1”. 


(d)
Answer : 
    lhs0 = lhs[i + 32 * k];
    rhs0 = rhs[k + 128 * j];
    lhs1 = lhs[i + 32 * (k + 1)];
    rhs1 = rhs[(k + 1) + 128 * j];
    O = output[i + 32 * j];
    FMA on lhs0, rhs0, O;
    FMA on lhs1, rhs1, O;
    Write O to output[i + 32 * j];

int i = threadIdx.x;
int j = threadIdx.y;
float temp1, temp2;
for (int k = 0; k < 128; k += 2) {
    temp1 = lhs[i + 32 * k] * rhs[k + 128 * j];
    temp2 = lhs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j];
    output[i + 32 * j] += temp1 + temp2;
}


(e)
Answer : 
    Increases k like processing 4 values of k rather than 2 by computing k, (k + 1), (k + 2), (k + 3) in one iteration. 

==============================================================================================
PART 2 - Matrix transpose optimization (65 points)
==============================================================================================