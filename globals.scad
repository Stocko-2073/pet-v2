$fn=0;
$fa=1;
$fs=$preview?0.5:0.25;

color1="#f8f8f8";
color2="#aaabaf";
color3="#88e0dd";

nozzle=0.4;
ep=0.01;
$slop=0.2;
d1=83;
x=68;
h1=x/2;
y=40;
//z1=sqrt(3/2)*256/3+x/4;
z1=175;
z2=93;
z3=z1+6;
wrist_gear_height=6;
axel_diameter=6;

num_assembly_steps=32;
scale_in_frames=32-26;

function map(val,fromLow,fromHigh,toLow,toHigh) = toLow + (val - fromLow) * (toHigh - toLow) / (fromHigh - fromLow);
function animate_keyframes(t,i,keyframes) = 
    let(tt=map(t,0,1,0,len(keyframes)-1)-0.000001)
    map(tt%1,0,1,keyframes[floor(tt)][i],keyframes[floor(tt)+1][i]);
function reverse(l) = [for(i=[len(l)-1:-1:0]) l[i]];

module asm(n) {
    if(is_undef(show_assembly)) {
        children();
    } else {
        assembly=floor(sin($t*180)*num_assembly_steps);
        if (assembly >= n+scale_in_frames) {
            children();
        } else if (assembly > n) {
            // Scale children based on fractional part between n and n+scale_in_frames
            scale_factor = (assembly-n)/scale_in_frames;
            s = 1 - pow(1-scale_factor,3); // Ease out cubic
            scale([s,s,s]) children();
        }
    }
}

module origin_point(color="red",d=10) {
    %color(color) sphere(d=d);
    if($children>0) children();
}

module nop() { children(); }

module skip() {}

module zview(flip=false,xray=false) {
    render() difference() {
        union() children();
        if (flip) {
            cuboid([300,300,300],anchor=BOT);
        } else {
            cuboid([300,300,300],anchor=TOP);
        }
    }
    if (xray) %render() difference() {
        union() children();
        if (!flip) {
            cuboid([300,300,300],anchor=BOT);
        } else {
            cuboid([300,300,300],anchor=TOP);
        }
    }
}

module xview(flip=false,xray=false) {
    render() difference() {
        union() children();
        if (flip) {
            cuboid([300,300,300],anchor=LEFT);
        } else {
            cuboid([300,300,300],anchor=RIGHT);
        }
    }
    if (xray) %render() difference() {
        union() children();
        if (!flip) {
            cuboid([300,300,300],anchor=LEFT);
        } else {
            cuboid([300,300,300],anchor=RIGHT);
        }
    }
}

module yview(flip=false,xray=false) {
    render() difference() {
        union() children();
        if (flip) {
            cuboid([300,100,300],anchor=FWD);
        } else {
            cuboid([300,100,300],anchor=BACK);
        }
    }
    if (xray) %render() difference() {
        union() children();
        if (!flip) {
            cuboid([300,100,300],anchor=FWD);
        } else {
            cuboid([300,100,300],anchor=BACK);
        }
    }
}

module servo_gear_socket(h=3.5,r=2.95) {
    difference() {
        union() down(nozzle*2) {
            cyl(r=r+0.35,h=h-nozzle*2,anchor=TOP);
            cyl(r1=r+0.35,d2=3,h=nozzle*2,anchor=BOT);
        }
        union() for (i=[0:24]) hull() {
            zrot(i*360/25) fwd(r) cyl(d=0.4,h=h,anchor=TOP,$fn=16);
            zrot(i*360/25) fwd(r+0.35) cyl(d=0.8,h=h,anchor=TOP,$fn=16);
        }
    }
}

