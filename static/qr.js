/* Minimal QR Code generator - alphanumeric mode for URLs */
var QR=(function(){
var P=function(){var a=[],b=[];a[0]=1;for(var c=1;c<256;c++){a[c]=a[c-1]<<1^(a[c-1]>>7?0x11d:0);b[a[c]]=c}return{a:a,b:b}}();
var E={L:1,M:0,Q:3,H:2};
var M=[[0,0,0,0],[0,1,0,1],[1,0,1,0],[1,1,1,1]];
var G={2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]};

function init(){
var cap={2:26,3:44,4:70,5:100,6:134,7:172,8:196,9:242,10:292};
var ec={2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]};
ec[2]=[10,7,15,11];ec[3]=[16,10,26,16];ec[4]=[26,15,18,26];
ec[5]=[18,20,20,18];ec[6]=[20,26,24,26];ec[7]=[24,18,16,30];
ec[8]=[30,20,18,22];ec[9]=[18,24,22,22];ec[10]=[20,28,26,24];
G={};for(var v=2;v<=10;v++){G[v]={cap:cap[v],ecb:ec[v][0],ecbl:ec[v][1],dcb:ec[v][2],dcbl:ec[v][3]}}
}

function getVersion(len){
init();
for(var v=2;v<=10;v++)if(len<=G[v].dcb)return v;
return 10;
}

function encode(text){
text=text.toUpperCase();
var len=text.length;
var ver=getVersion(Math.ceil(len*1.2));
var total=G[ver].dcb+G[ver].dcbl;
var bits=0;
for(var i=0;i<text.length;i++)bits+=getCharBits(text[i]);

var data="";
data+="0100";
var lenBits=ver<=9?8:16;
data+=toBin(len,lenBits);
for(var i=0;i<text.length;i++){
var ci=getCharIndex(text[i]);
var nb=getCharBits(text[i]);
data+=toBin(ci,nb);
}
while(data.length<total*8)data+="0";
if(data.length%8)data+="0".repeat(8-data.length%8);
return {data:data,version:ver};
}

function getCharBits(c){
if(c>='0'&&c<='9')return 4;
if(c>='A'&&c<='Z')return 6;
if(c===' ')return 4;
return 8;
}

function getCharIndex(c){
if(c>='0'&&c<='9')return parseInt(c);
if(c>='A'&&c<='Z')return parseInt(c,36)-10;
if(c===' ')return 36;
return 0;
}

function toBin(n,len){
var s="";
for(var i=len-1;i>=0;i--)s+=((n>>i)&1)?"1":"0";
return s;
}

function addECC(data,ver){
init();
var g=G[ver];
var ecCount=g.ecb+g.ecbl;
var dataCount=g.dcb+g.dcbl;
var totalBytes=dataCount+ecCount;
var blocks=[];
var offset=0;
var blockSize=Math.floor(dataCount/g.ecbl);
var longBlocks=dataCount%g.ecbl;

for(var i=0;i<g.ecbl;i++){
var sz=blockSize+(i>=g.ecbl-longBlocks?0:1);
blocks.push(data.substr(offset*8,sz*8));
offset+=sz;
}

var ecBlocks=[];
for(var i=0;i<g.ecbl;i++){
var block=blocks[i];
var bytes=[];
for(var j=0;j<block.length;j+=8)bytes.push(parseInt(block.substr(j,8),2));
var ec=reedSolomon(bytes,ecCount);
var ecStr="";
for(var j=0;j<ec.length;j++)ecStr+=toBin(ec[j],8);
ecBlocks.push(ecStr);
}

var result="";
var maxDataLen=0;
for(var i=0;i<blocks.length;i++)if(blocks[i].length>maxDataLen)maxDataLen=blocks[i].length;

for(var i=0;i<maxDataLen;i+=8){
for(var j=0;j<g.ecbl;j++){
if(i<blocks[j].length)result+=blocks[j].substr(i,8);
}
}
for(var i=0;i<ecCount*8;i+=8){
for(var j=0;j<g.ecbl;j++){
if(i<ecBlocks[j].length)result+=ecBlocks[j].substr(i,8);
}
}
return result;
}

function reedSolomon(data,ecLen){
var gen=rsGenPoly(ecLen);
var rem=new Array(ecLen).fill(0);
for(var i=0;i<data.length;i++){
var coef=data[i]^rem.shift();
rem.push(0);
if(coef!==0){
for(var j=0;j<ecLen;j++){
rem[j]^=P.a[(P.b[coef]+P.b[gen[j+1]])%255];
}
}
}
return rem;
}

function rsGenPoly(degree){
var poly=[1];
for(var i=0;i<degree;i++){
var newPoly=new Array(poly.length+1).fill(0);
for(var j=0;j<poly.length;j++){
newPoly[j]^=poly[j];
newPoly[j+1]^=P.a[(P.b[poly[j]]+i)%255];
}
poly=newPoly;
}
return poly;
}

function placeModule(matrix,size,module,x,y){
if(x>=0&&x<size&&y>=0&&y<size&&matrix[y][x]===-1){
matrix[y][x]=module?1:0;
}
}

function generate(text){
init();
var enc=encode(text);
var ver=enc.version;
var data=enc.data;
var size=ver*4+17;
var matrix=Array.from({length:size},()=>Array(size).fill(-1));
var reserved=Array.from({length:size},()=>Array(size).fill(false));

// Finder patterns
for(var r=0;r<7;r++)for(var c=0;c<7;c++){
var v=(r===0||r===6||c===0||c===6)||(r>=2&&r<=4&&c>=2&&c<=4);
placeModule(matrix,size,v,r,c);reserved[r][c]=true;
}
for(var i=0;i<8;i++){
placeModule(matrix,size,false,7+i,0);reserved[0][7+i]=true;
placeModule(matrix,size,false,0,7+i);reserved[7+i][0]=true;
placeModule(matrix,size,false,i,size-8);reserved[size-8+i][size-1]=true;
placeModule(matrix,size,false,size-8+i,size-8);reserved[size-8][size-8+i]=true;
}

// Timing patterns
for(var i=8;i<size-8;i++){
placeModule(matrix,size,i%2===0,6,i);reserved[6][i]=true;
placeModule(matrix,size,i%2===0,i,6);reserved[i][6]=true;
}

// Alignment patterns
if(ver>=2){
var alignPos=[6,18,22,26,30,34,38,42,46];
var count=ver>=7?Math.floor(ver/7)+2:Math.floor(ver/7)+1;
var positions=[];
for(var i=0;i<count;i++)positions.push(alignPos[i]);
for(var r=0;r<positions.length;r++)for(var c=0;c<positions.length;c++){
var y=positions[r],x=positions[c];
if(reserved[y]&&reserved[y][x])continue;
for(var dr=-2;dr<=2;dr++)for(var dc=-2;dc<=2;dc++){
var v=Math.abs(dr)===2||Math.abs(dc)===2||(dr===0&&dc===0);
placeModule(matrix,size,v,y+dr,x+dc);reserved[y+dr][x+dc]=true;
}
}
}

// Dark module
placeModule(matrix,size,true,size-8,8);reserved[size-8][8]=true;

// Reserve format info areas
for(var i=0;i<15;i++){
var y=i<6?i:i<8?i+1:i<15?i-1:0;
reserved[8][i<8?i:i+1]=true;
reserved[y<8?y:y+1][8]=true;
}

// Add ECC
var finalData=addECC(data,ver);

// Place data
var bitIdx=0;
for(var right=size-1;right>=1;right-=2){
if(right===6)right=5;
for(var vert=0;vert<size;vert++){
for(var j=0;j<2;j++){
var x=right-j;
var y=((right+1)&2)===0?vert:size-1-vert;
if(!reserved[y][x]&&bitIdx<finalData.length*8){
var byteIdx=Math.floor(bitIdx/8);
var bitPos=7-bitIdx%8;
var module=((finalData.charCodeAt(byteIdx)-(finalData.charCodeAt(byteIdx)>191?848:48))>>bitPos)&1;
placeModule(matrix,size,module===1,y,x);
bitIdx++;
}
}
}
}

// Apply mask 0 (checkerboard)
var masked=Array.from({length:size},(_,r)=>Array.from(matrix[r],(v,c)=>{
if(reserved[r][c])return v;
return v===-1?((r+c)%2===0?1:0):v^(((r+c)%2===0)?1:0);
}));

// Format info (mask 0, error correction M)
var formatBits=[0,1,0,0,1,0,0,0,0,0,1,0,0,1,0];
var format=0;
for(var i=0;i<15;i++){
if(formatBits[i])format|=(1<<i);
}
format^=0x5412;

for(var i=0;i<6;i++)masked[8][i]=(format>>(14-i))&1;
masked[8][7]=(format>>8)&1;
masked[8][8]=(format>>7)&1;
masked[7][8]=(format>>6)&1;
for(var i=0;i<6;i++)masked[5-i][8]=(format>>(i))&1;

return {matrix:masked,size:size};
}

return {generate:generate};
})();
