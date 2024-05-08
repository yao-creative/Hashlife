import time
import math
import weakref
#import numpy as np
#import math
import weakref
GROUP = [ 
    'yi-yao.tan@polytechnique.edu', 
    'sameh.ammari@polytechnique.edu'
    ]
HC = weakref.WeakValueDictionary()

def calc_round(num_cols,num_rows,cells):
    """Using a grid of cells """
    new = [[False for _ in range(num_cols)] for _ in range(num_rows)]
    for k,row in enumerate(cells):
        for h,cell in enumerate(row):
            adjacent = 0
            for i in range(-1,2):
                for j in range(-1,2):
                    if (i==0 and j==0):
                        continue
                    if i+k == num_rows or i+k<0 or j+h==num_cols or j+h<0:
                        continue
                    elif cells[k+i][h+j]: 
                        adjacent +=1
                    if adjacent>3:
                        break
            if adjacent== 3:
                new[k][h] = True
            elif cell and adjacent == 2:
                new[k][h] = True
            else:
                new[k][h] = False
    return new
def unpack_lvl2node(node):
    """Takes a level 2 node and unpacks it into a 4x4 grid"""
    #print(f"node.level: {node.level}")
    grid = [[False for _ in range(4)] for _ in range(4)]
    grid[0][0] = node.nw.nw.alive
    grid[0][1] = node.nw.ne.alive
    grid[0][2] = node.ne.nw.alive
    grid[0][3] = node.ne.ne.alive
    grid[1][0] = node.nw.sw.alive
    grid[1][1] = node.nw.se.alive
    grid[1][2] = node.ne.sw.alive
    grid[1][3] = node.ne.se.alive
    grid[2][0] = node.sw.nw.alive
    grid[2][1] = node.sw.ne.alive
    grid[2][2] = node.se.nw.alive
    grid[2][3] = node.se.ne.alive
    grid[3][0] = node.sw.sw.alive
    grid[3][1] = node.sw.se.alive
    grid[3][2] = node.se.sw.alive
    grid[3][3] = node.se.se.alive
    #print(f"unpacked grid: {grid}")
    return grid

def pack_lvl2node(grid):
    """Takes a grid and packs it into a level 2 node (updating only center 2x2)"""
    #print(f"grid to pack: {grid}")
    out =AbstractNode.node(nw= AbstractNode.cell(grid[1][1]), ne = AbstractNode.cell(grid[1][2]), sw = AbstractNode.cell(grid[2][1]), se= AbstractNode.cell(grid[2][2]))
    #print(f"out:{repr(out)}")
    return out
    
def node_to_mask(node):
    """Turns a mask 16 bit of 4x4 grid from bottom to top row by row"""
    out = 0
    if node.se.se.alive:
        out += 1
    if node.se.sw.alive:
        out += 2
    if node.sw.se.alive:
        out+= 2**2
    if node.sw.sw.alive:
        out+= 2**3
    if node.se.ne.alive:
        out += 2**4
    if node.se.nw.alive:
        out +=2**5
    if node.sw.ne.alive:
        out += 2**6
    if node.sw.nw.alive:
        out += 2**7
    if node.ne.se.alive:
        out += 2**8
    if node.ne.sw.alive:
        out += 2**9
    if node.nw.se.alive:
        out += 2**10
    if node.nw.sw.alive:
        out += 2**11
    if node.ne.ne.alive:
        out += 2**12
    if node.ne.nw.alive:
        out += 2**13
    if node.nw.ne.alive:
        out += 2**14
    if node.nw.nw.alive:
        out += 2**15
    return out
def calc_round_bitmasking(mask):
    #print(f"mask: {mask}")
    """Using the bit masking algorithm we count the number of ones 
    Then out put a 4x4 grid with the middle 2x2 updated and periphral of falses """
    set_E5 = [0,1,2,4,6,8,9,10]
    set_E6 = [i+1 for i in set_E5]
    set_E9 = [4,5,6,8,10,12,13,14]
    set_E10 = [i+1 for i in set_E9]
    #mapping of the specific indices of cells and their set of indices in the bool arr
    l =[(5,set_E5),(6,set_E6),(9,set_E9),(10,set_E10)]
    grid = [[False for _ in range(4)] for _ in range(4)]
    for idx, neighbors in l:
        #print(f"idx: {idx}")
        h = idx//4 #the row of focus cell from the bottom
        k = idx%4
        #print(f"h: {h} k:{k} neighbors: {neighbors}")
        N = 0

        N = sum(((mask>>i)%2)<<i for i in neighbors)
        #turn the specific neighbors into byte
        if mask ==0: #all dead nothing to calculate
            continue
        i = 0
        #print(f"idx: {idx} N: {N}")
        while i <=4 and N!=0:#loop until we get 2^8 -1 then go back one step and # of steps = # digits with 1
            N = N&(N-1)
            i+=1
            if N == 0:
                break
        #print(f"i: {i}")
        #print(f"current state: {cur_state}")
        if ((i ==2 or i==3) and ((mask >> idx)% 2 == 1)) or ((((mask >> idx)% 2) == 0) and i==3):
            grid[3-h][3-k] = True #set new grid values
    return grid 
def check_periph(node):
    """Checks if there is a peripheral band which might disappear"""
    if node.level>=2:
        center = AbstractNode.node(node.nw.se,node.ne.sw, node.sw.ne, node.se.nw)
        if node.population == center.population: 
            #if there's only 2 nodes in peripheral it will never be able to escape in 2^k-2 time
            #root size not big enough to calculate future so we extend
            #print(f"peripheral!!")
            return False
    return True
class Universe:
    def round(self):
        """Compute (in place) the next generation of the universe"""
        raise  

    def get(self, i, j):
        """Returns the state of the cell at coordinates (ij[0], ij[1])"""
        raise NotImplementedError

    def rounds(self, n):
        """Compute (in place) the n-th next generation of the universe"""
        for _i in range(n):
            self.round()
    def __repr__(self):
        out = ""
        for row in self.cells:
            out+=("__"*4*len(self.cells)+"\n")
            string = "|"
            for cell in row:
                if cell:
                    string += "X|"
                else:
                    string += "O|"
            out+=(string+ "\n")
        out+=("__"*4*len(self.cells)+"\n")
        return out


class HashLifeUniverse(Universe):
    def __init__(self, *args):
        if len(args) == 1:
            self._root = args[0]
        else:
            self._root = HashLifeUniverse.load(*args)

        self._generation = 0
        
    @staticmethod
    def load(n, m, cells):
        level = math.ceil(math.log(max(1, n, m), 2))

        mkcell = getattr(AbstractNode, 'cell', CellNode)
        mknode = getattr(AbstractNode, 'node', Node    )

        def get(i, j):
            i, j = i + n // 2, j + m // 2
            return \
                i in range(n) and \
                j in range(m) and \
                cells[i][j]
                
        def create(i, j, level):
            if level == 0:
                return mkcell(get (i, j))

            noffset = 1 if level < 2 else 1 << (level - 2)
            poffset = 0 if level < 2 else 1 << (level - 2)

            nw = create(i-noffset, j+poffset, level - 1)
            sw = create(i-noffset, j-noffset, level - 1)
            ne = create(i+poffset, j+poffset, level - 1)
            se = create(i+poffset, j-noffset, level - 1)

            return mknode(nw=nw, ne=ne, sw=sw, se=se)
                
        return create(0, 0, level)

    def get(self, i, j):
        """Recursively go to search the state of the cell at coordinates (i,j)"""
        return self._root.get(i,j)

    def rounds(self, n):
        # Do something here
        i=0
        #print(f"self._root.level {self._root.level}")
        while n !=0:
            if n%2 ==1:
                #if The universe is too small and things could potentially leave
                #extend it if needed
                #we need level k to compute k-2 so we ensure level is at least k
                if self._root.level<i+2:
                    self.extend(i+2)
                #if there's peripheral we extend to encompass the peripheral in the center
                if check_periph(self._root):
                    self.extend(self._root.level+1)
                #after the previous two if statements 
                #set ourself up for the forward by centeralizing our future outcome.
                self.extend(self._root.level+1)
                self._root = self._root.forward(i)
                self._generation+=2^i
            i+=1
            n>>=1
    
            

    def round(self):
        return self.rounds(1)

    @property
    def root(self):
        return self._root
        
    @property
    def generation(self):
        return self._generation

    def extend(self, k):
        """extends the enclosed quadtree until the peripheral
        band is empty and its level (parameter .level) 
        is at least equal to max(k, 2)"""
        periph = check_periph(self._root)
        #if there are peripheral nodes or root level is less than the level we want or k
        #we extend until we can fit alive universe inside.
        while self._root.level<max(2,k):
            self._root = self._root.extend()
        
        #in the end original universe will be in the center and periphral will be empty

class NaiveUniverse(Universe):
    def __init__(self, n, m, cells):
        """cells: a grid of all the cells"""
        self.num_rows=n
        self.num_columns=m
        self.cells= cells
    def round(self):
        """Updates in place universe and list of cells that are alive"""
        self.cells = calc_round(self.num_columns,self.num_rows,self.cells)
    def get(self, i, j):
        """returns whether the cells with coordinates (i,j) is alive"""
        return self.cells[i][j]

class AbstractNode:
    BIG = True
    def __init__(self):
        self._cache = None
        self._hash = None

    def get(self,i,j):
        #the whole grid is shifted up and to the right by one block
        if (i>2**(self.level-1)-1 or i<-2**(self.level-1) or j>2**(self.level-1)-1 or j<-2**(self.level-1)) and self.level !=0:
            return False
        #If we hit the bottom we just return the result
        if self.level == 0:
            return self.alive
        #shift coordinates to recentralize them in each quadrant
        elif i>=0 and j>=0:
            return self.ne.get(i-2**(self.level-2), j-2**(self.level-2))
        elif i<0 and j>=0:
            return self.nw.get(i+2**(self.level-2), j-2**(self.level-2))
        elif i>=0 and j<0:
            return self.se.get(i-2**(self.level-2), j+2**(self.level-2))
        else: # i<0 and j>0
            return self.sw.get(i+2**(self.level-2), j+2**(self.level-2))
    def __hash__(self):
        """Modified hash function that only looks at structure"""
        if self._hash is None:
            self._hash = (
                self.population,
                self.level     ,
                self.nw        ,
                self.ne        ,
                self.sw        ,
                self.se        ,
            )
            self._hash = hash(self._hash)
        return self._hash
    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, AbstractNode):
            return False
        return \
            self.level      == other.level      and \
            self.population == other.population and \
            self.nw         is other.nw         and \
            self.ne         is other.ne         and \
            self.sw         is other.sw         and \
            self.se         is other.se
    def __repr__(self):
        return '{}({}, {}, {}, {}) level: {}'.format(
                 self.__class__.__name__,
                 repr(self.nw),
                 repr(self.ne),
                 repr(self.sw),
                 repr(self.se),
                 self.level)
    @property
    def level(self,k):
        """Level of this node"""
        raise NotImplementedError()

    @property
    def population(self):
        """Total population of the area"""
        raise NotImplementedError()

    nw = property(lambda self : None)
    ne = property(lambda self : None)
    sw = property(lambda self : None)
    se = property(lambda self : None)

    @staticmethod
    def zero(k):
        """Create level 2^{2k} amount of zeros and place them in a square"""
        dead_cell = AbstractNode.cell(False)
        
        if k == 0:
            return  dead_cell
        elif k >= 1:
            nw, ne, sw, se = dead_cell,dead_cell,dead_cell,dead_cell
            out = AbstractNode.node(nw,ne,sw,se)
            while k>1:
                a = AbstractNode.node(nw,ne,sw,se)
                nw = a
                ne = a
                sw = a
                se = a
                out = AbstractNode.node(nw,ne,sw,se)
                k-=1
        return out

     
            
    def extend(self):
        """Extend the cell by a factor of 4 and place the current node in the nw"""
        if self.level == 0:
            dead_cell = AbstractNode.cell(False)
            nw = AbstractNode.cell(self.alive)
            ne,sw,se = dead_cell,dead_cell,dead_cell
        else:
            z =self.zero(self.level-1)
            nw = AbstractNode.node(nw =z ,ne= z,sw= z,se =self.nw)
            se = AbstractNode.node(nw= self.se,ne= z,sw = z,se= z)
            ne = AbstractNode.node(nw =z,ne= z, sw=self.ne,se =z)
            sw = AbstractNode.node(nw =z,ne= self.sw,sw=z,se =z)
        return AbstractNode.node(nw,ne,sw,se) #and everything else around it)
    def forward(self, l =None):
        """Calculate the state of the center 2^{k-2} time steps in the future"""
        if self._cache ==None:
            #make a list for the cache to account for different forward l possibilities
            self._cache = [None for _ in range(self.level-1)]
        #print(f"l: {l} level: {self.level-2}")
        if self.level<2:
            return None
        if l is None and self._cache[-1] is not None:
            #return self.lvl-2 future
            return self._cache[-1]
        elif l is not None and self._cache[l] is not None:
            return self._cache[l]
        if self.population ==0: #optimization 0 population, nothing changes
            return AbstractNode.zero(self.level-1)
        if self.level == 2:
            #print(f"self.lvl: {self.level}")
            mask = node_to_mask(self)
            return Node.level2_bitmask(mask)
        if l ==None or l >= self.level-2:
            R_nw,R_ne,R_sw,R_se= self.nw.forward(),self.ne.forward(),self.sw.forward(),self.se.forward()
            #compute this in l generations in the future
            R_TC =  AbstractNode.node(nw =self.nw.ne, ne =self.ne.nw, sw= self.nw.se, se = self.ne.sw).forward()
            R_CL =  AbstractNode.node(nw =self.nw.sw, ne =self.nw.se, sw= self.sw.nw, se = self.sw.ne).forward()
            R_CC =  AbstractNode.node(nw =self.nw.se, ne =self.ne.sw, sw= self.sw.ne, se = self.se.nw).forward()
            R_BC =  AbstractNode.node(nw =self.sw.ne, ne =self.se.nw, sw= self.sw.se, se = self.se.sw).forward()
            R_CR =  AbstractNode.node(nw =self.ne.sw, ne =self.ne.se, sw= self.se.nw, se = self.se.ne).forward()
            #we get l generations in the future
            A_nw = AbstractNode.node(nw = R_nw,ne = R_TC, sw = R_CL, se= R_CC)
            A_ne = AbstractNode.node(nw = R_TC,ne = R_ne, sw = R_CC, se= R_CR)
            A_sw = AbstractNode.node(nw = R_CL,ne = R_CC, sw = R_sw, se= R_BC)
            A_se = AbstractNode.node(nw = R_CC,ne = R_CR, sw = R_BC, se= R_se)
            B_nw= A_nw.forward()
            B_ne= A_ne.forward()
            B_sw= A_sw.forward()
            B_se= A_se.forward()
            if l is not None:
                self._cache[l] = AbstractNode.node(nw = B_nw, ne = B_ne, sw= B_sw, se = B_se)
                return self._cache[l]
            else:
                self._cache[-1] = AbstractNode.node(nw = B_nw, ne = B_ne, sw= B_sw, se = B_se)
                return self._cache[-1]
        else: #if l < k-2 
            #l maintain the state we have and construct the nodes upwards.
            R_nw,R_ne,R_sw,R_se= self.nw.forward(l),self.ne.forward(l),self.sw.forward(l),self.se.forward(l)
            #compute this in l generations in the future
            R_TC =  AbstractNode.node(nw =self.nw.ne, ne =self.ne.nw, sw= self.nw.se, se = self.ne.sw).forward(l)
            R_CL =  AbstractNode.node(nw =self.nw.sw, ne =self.nw.se, sw= self.sw.nw, se = self.sw.ne).forward(l)
            R_CC =  AbstractNode.node(nw =self.nw.se, ne =self.ne.sw, sw= self.sw.ne, se = self.se.nw).forward(l)
            R_BC =  AbstractNode.node(nw =self.sw.ne, ne =self.se.nw, sw= self.sw.se, se = self.se.sw).forward(l)
            R_CR =  AbstractNode.node(nw =self.ne.sw, ne =self.ne.se, sw= self.se.nw, se = self.se.ne).forward(l)
            nw = AbstractNode.node(nw = R_nw.se, ne= R_TC.sw, sw= R_CL.ne, se = R_CC.nw)
            ne = AbstractNode.node(nw = R_TC.se, ne= R_ne.sw, sw= R_CC.ne, se = R_CR.nw)
            sw = AbstractNode.node(nw = R_CL.se, ne= R_CC.sw, sw= R_sw.ne, se = R_BC.nw)
            se = AbstractNode.node(nw = R_CC.se, ne= R_CR.sw, sw= R_BC.ne, se = R_se.nw)
            self._cache[l] =AbstractNode.node(nw = nw, ne = ne, sw= sw, se = se)
        return self._cache[l]
    def naive_forward(self):
        """Calculate the state of the center 2^{k-2} time steps in the future"""
        #print(f"self.level: {self.level}")
        
        if self._cache is not None:
            return self._cache
        if self.level<2:
            return None
        if self.population ==0: #optimization 0 population, nothing changes
            return AbstractNode.zero(self.level-1)
            #0.009 seconds with this optimisation and about 0.016 seconds run time without(including printing the dictionary)
        if self.level == 2:
            #unpack the cells and use the naive algorithm to calculate the state of the level2 node after a round.
            #print(f"naive algo")
            grid = unpack_lvl2node(self)
            grid = calc_round(4,4,grid)
            return pack_lvl2node(grid)
        
        #print(f"Running R's")
        R_nw,R_ne,R_sw,R_se= self.nw.naive_forward(),self.ne.naive_forward(),self.sw.naive_forward(),self.se.naive_forward()

        #print(f"R_sw: {R_sw}")

        R_TC =  AbstractNode.node(nw =self.nw.ne, ne =self.ne.nw, sw= self.nw.se, se = self.ne.sw).naive_forward()
        R_CL =  AbstractNode.node(nw =self.nw.sw, ne =self.nw.se, sw= self.sw.nw, se = self.sw.ne).naive_forward()
        R_CC =  AbstractNode.node(nw =self.nw.se, ne =self.ne.sw, sw= self.sw.ne, se = self.se.nw).naive_forward()
        R_BC =  AbstractNode.node(nw =self.sw.ne, ne =self.se.nw, sw= self.sw.se, se = self.se.sw).naive_forward()
        R_CR =  AbstractNode.node(nw =self.ne.sw, ne =self.ne.se, sw= self.se.nw, se = self.se.ne).naive_forward()


        A_nw = AbstractNode.node(nw = R_nw,ne = R_TC, sw = R_CL, se= R_CC)
        A_ne = AbstractNode.node(nw = R_TC,ne = R_ne, sw = R_CC, se= R_CR)
        A_sw = AbstractNode.node(nw = R_CL,ne = R_CC, sw = R_sw, se= R_BC)
        A_se = AbstractNode.node(nw = R_CC,ne = R_CR, sw = R_BC, se= R_se)


        B_nw= A_nw.naive_forward()
        B_ne= A_ne.naive_forward()
        B_sw= A_sw.naive_forward()
        B_se= A_se.naive_forward()

        
        self._cache = AbstractNode.node(nw = B_nw, ne = B_ne, sw= B_sw, se = B_se)
        return self._cache
    @staticmethod
    def canon(node):
        """Canonical representative of each node using their hash"""
        return HC.setdefault(node,node)

    @staticmethod
    def cell(alive):
        return AbstractNode.canon(CellNode(alive)) 

    @staticmethod
    def node(nw,ne,sw,se):
        return AbstractNode.canon(Node(nw,ne,sw,se))
    
class CellNode(AbstractNode):
    def __init__(self, alive):
        super().__init__()

        self._alive = bool(alive)

    level      = property(lambda self : 0)
    population = property(lambda self : int(self._alive))
    alive      = property(lambda self : self._alive)
    def __repr__(self):
        return f"Cell({self.alive})"
class Node(AbstractNode):
    def __init__(self, nw, ne, sw, se):
        super().__init__()
        self._level      = 1 + nw.level
        self._population =  \
            nw.population + \
            ne.population + \
            sw.population + \
            se.population
        self._nw = nw
        self._ne = ne
        self._sw = sw
        self._se = se
    @staticmethod
    def level2_bitmask(mask):
        grid = calc_round_bitmasking(mask)   
        return pack_lvl2node(grid)

    level      = property(lambda self : self._level)
    population = property(lambda self : self._population)

    nw = property(lambda self : self._nw)
    ne = property(lambda self : self._ne)
    sw = property(lambda self : self._sw)
    se = property(lambda self : self._se)
"""
BIT MASKING TEST
-------------------------------------------------------
print(f"bit masking: {Node.level2_bitmask(14592)}")
"""

#data = (36, 9, [[False, False, False, True, True, False, False, False, False], [False, False, False, True, True, False, False, False, False], [False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], [False, False, True, True, True, False, False, False, False], [False, True, False, False, False, True, False, False, False], [True, False, False, False, False, False, True, False, False], [True, False, False, False, False, False, True, False, False], [False, False, False, True, False, False, False, False, False], [False, True, False, False, False, True, False, False, False], [False, False, True, True, True, False, False, False, False], [False, False, False, True, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], [False, False, False, False, True, True, True, False, False], [False, False, False, False, True, True, True, False, False], [False, False, False, True, False, False, False, True, False], [False, False, False, False, False, False, False, False, False], [False, False, True, True, False, False, False, True, True], [False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], [False, False, False, False, False, True, True, False, False], [False, False, False, False, False, True, True, False, False]])
#data  =(3, 1, [[True], [True], [True]])


"""gen, m, n, data = ([0, 1, 2, 2, 1, 5, 4, 4], 3, 1, [[True], [True], [True]])
if isinstance(gen, int):
  gen = [None] * gen

node = HashLifeUniverse(m, n, data).root

for g in gen:
    for _ in range(2):         # see the log
        node = node.extend()
node = node.forward(g) if g is not None else node.forward()"""
"""
INITIATE UNIVERSE:
-----------------------------------------------------"""
"""U1= HashLifeUniverse(*data)
U1.rounds(5)
print(f"U1.root: {U1.root}")"""

"""
#print(f"node: {node}")

TEST GET
-------------------------------------
print(f"U1.get(-1,0): {U1.get(-1,0)}")
print(f"U1.get(0,0): {U1.get(0,0)}")
print(f"U1.get(1,0): {U1.get(1,0)}")
print(f"U1.get(2,0): {U1.get(2,0)}")
print(f"U1.get(3,0): {U1.get(3,0)}")
print(f"U1.get(0,-1): {U1.get(0,-1)}")
print(f"U1.get(0,1): {U1.get(0,1)}")"""

"""
EXTEND UNIVERSE:
-------------------------------------

U1.extend(3)
print(f"U1.extend(3): {U1.root}")"""


"""
NAIVE VS BITMASKING SPEED TEST
----------------------------------------------------
t1 = time.time()
node= U1.root
n =node.extend().extend()
n.forward()
print(f"forward time: {time.time() - t1}")

nodes_dict= dict() #empty the dictionary
node2 = U1.root
n2 = node.extend().extend()
t2 = time.time()
n2.naive_forward()
print(f"naive_forward time: {time.time() - t2}") #naive forward on average takes 0.001 seconds longer
or 25% percent longer maybe due to the methods I pack the grid in"""
