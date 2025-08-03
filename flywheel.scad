include <BOSL2/std.scad>
include <MCAD/servos.scad>
include <globals.scad>
include <BOSL2/screws.scad>

module servo_9g() {
    %towerprosg90([0,0,26.7],[180,0,0],0,0,1);
    down(5) children();
}

module servo9g_gear_socket(h=3.5) {
    $fn=16;
    union() {
        r=4.9/2;
        difference() {
            cyl(r=r+0.35,h=h, anchor=TOP);
            union() for (i=[0:20]) hull() {
                zrot(i*360/21) fwd(r) cyl(d=0.4,h=h, anchor=TOP);
                zrot(i*360/21) fwd(r+0.35) cyl(d=0.8,h=h, anchor=TOP);
            }
        }
    }
}

module as5600() {
    down(1.5) {
        color("#444") zview() translate([93.5,-104,-10.5]) import("AS5600.stl");
        color("#eee") down(1.5) zview() up(1.5) zview(true) translate([93.5,-104,-10.5]) import("AS5600.stl");
        color("#444") down(1.5) zview(true) up(1.5) translate([93.5,-104,-10.5]) import("AS5600.stl");
    }
}
*!as5600();

module servo_9g_neg($slop=0.2) {
    union() {
        up(4) {
            zrot(180) servo_9g_top_neg();
            left(5.9+$slop) down($slop) cuboid([22.5+$slop*2,11.8+$slop*2,22.7+$slop*2],anchor=BOT+LEFT);
            down($slop) left(5.9+$slop+4.7) up(4.3) cuboid([32.1+$slop*2,11.8+$slop*2,2.5+$slop*2],anchor=BOT+LEFT);
            // Wires
            up(22.7+$slop) left(5.9) cuboid([4.7,11.8+$slop*2,10],anchor=TOP+RIGHT);
            up(22.7+$slop) left(5.9) cuboid([4.7,4,18],anchor=TOP+RIGHT);
        }
    }
}

flywheelMass=0.05; // kg
flywheelDensity=1220; // kg/m^3
flywheelRadius=0.04; // meters

flywheelLength=flywheelMass/(PI*flywheelRadius^2*flywheelDensity);

wall=2;


r=flywheelRadius*1000; // mm
l=flywheelLength*1000; // mm
t=abs(sin($t*360));

xrot(90) {
    render() {
    color("#588")
    difference() {
        cyl(r=r,l=l,anchor=BOT);
        //xflip_copy() left(12.5) down(ep) cyl(d=1.5,l=l+ep*2,anchor=BOT);
        up(l+ep)cyl(d=4+$slop+wall*2+$slop,l=4.67-1.5,anchor=TOP);
        up(3.5) servo9g_gear_socket();
        down(ep)cyl(d=2,l=l+ep*2,anchor=BOT);
    }

    up(t*l*2+$slop/2)
    !xrot(180) // Print
    color("#858")
    difference() {
        d=4+wall*2+$slop*2;
        union() {
            up(l) cyl(d=d,l=4.67-2.5,anchor=TOP);
            up(l) cyl(d=d+wall*2,l=wall,anchor=BOT);
        }
        down(ep) cyl(d=2,l=l+ep*2,anchor=BOT);
        up(l+wall+ep) hull() {
            cyl(d=4+$slop,l=wall+ep,anchor=TOP);
            cyl(d=2,l=wall+ep+0.25,anchor=TOP);
        }
    }
    }

    up(t*l*3) color("#888") up(l) cyl(d=4,l=2,anchor=BOT);
    down(1.6) import("sg_90_Horns.stl");
    down(5.5) zrot(90) xrot(180) {

        //!xrot(-90) // print
        color("#585") difference() {
            right(30) fwd(6) up(4) cuboid([75,25,25],anchor=BOT+RIGHT+FWD);
            servo_9g_neg();

            left(43) up(16.5) back(6) {
                yflip_copy()
                zflip_copy()
                fwd(6.5) up(7.55)
                yrot(90) zrot(-90) {
                    nut_trap_side(trap_width=12,"M3");
                    down(4) screw_hole("M3,15",head="button",atype="head",anchor=BOT,orient=BOT,counterbore=10);
                }
            }
            fwd(4) up(16) right(24) left(20) xflip_copy() right(20) yrot(-90) xrot(-90) {
                nut_trap_side(trap_width=20,"M3");
                down(4) screw_hole("M3,15",head="button",atype="head",anchor=BOT,orient=BOT,counterbore=10);
            }
        }

        servo_9g();

        //!xrot(90) // print
        color("#885") difference() {
            right(30) fwd(6+$slop) up(4) cuboid([52,2,20],anchor=BOT+RIGHT+BACK);
            fwd(4+$slop) up(16) right(24) left(20) xflip_copy() right(20) yrot(-90) xrot(-90) {
                nut_trap_side(trap_width=20,"M3");
                down(4) screw_hole("M3,15",head="button",atype="head",anchor=BOT,orient=BOT,counterbore=10);
            }
        }

        //!yrot(90) // print
        color("#558") difference() {
            down(9.5) left(45+$slop) cuboid([5,80,80],anchor=RIGHT);
            left(43) up(16.5) back(6) {
                yflip_copy()
                zflip_copy()
                fwd(6.5) up(7.55)
                yrot(90) zrot(-90) {
                    nut_trap_side(trap_width=12,"M3");
                    down(4) screw_hole("M3,15",head="button",atype="head",anchor=BOT,orient=BOT,counterbore=10);
                }
            }
            left(43) down(37) {
                yflip_copy()
                zflip_copy()
                fwd(6.5)
                up(7.55)
                yrot(90) zrot(-90) {
                    nut_trap_side(trap_width=12,"M3");
                    down(4) screw_hole("M3,15",head="button",atype="head",anchor=BOT,orient=BOT,counterbore=10);
                }
            }
        }

        //!xrot(90) // print
        color("#855") difference() {
            left(45) down(49.5) cuboid([25,29,25],anchor=BOT+LEFT);
            left(22) down(40) xrot(180) yrot(-90){
                nut_trap_side(trap_width=20,"M3");
                down(4) screw_hole("M3,15",head="button",atype="head",anchor=BOT,orient=BOT,counterbore=10);
            }
            left(43) down(37) {
                yflip_copy()
                zflip_copy()
                fwd(6.5)
                up(7.55)
                yrot(90) zrot(-90) {
                    nut_trap_side(trap_width=12,"M3");
                    down(4) screw_hole("M3,15",head="button",atype="head",anchor=BOT,orient=BOT,counterbore=10);
                }
            }
        }

        //!yrot(-90) // print
        color("#585") difference() {
            union() {
                left(20-$slop) down(49.5) cuboid([7.5,29,30.15],anchor=BOT+LEFT);
                zrot(90) xflip_copy() left(25/2+2) down(44.15+$slop) cuboid([9,25,25],anchor=BOT+LEFT);
            }
            yflip_copy() xflip_copy()
            right(8) down(21.65) fwd(8) zrot(180) yrot(180) zrot(90) {
                up(7) nut_trap_side(trap_width=12,"M3");
                down(4) screw_hole("M3,15",head="button",atype="head",anchor=BOT,orient=BOT,counterbore=10);
            }
            left(22) down(40) xrot(180) yrot(-90){
                nut_trap_side(trap_width=20,"M3");
                for(i=[-3:0.5:3]) left(i) down(4) screw_hole("M3,15",head="button",atype="head",anchor=BOT,orient=BOT,counterbore=10);
            }
        }
    }
    xflip_copy() left(12.5) down(2) {
        %screw("M1.4,10",head="socket",atype="head",anchor=BOT,orient=BOT);
    }
    up(l+wall+0.5) zrot(90) xrot(180) as5600();
}
echo("flywheelLength:",flywheelLength);

echo(str("\n",
"sh ./do_mp4.sh flywheel.scad ",
$vpt[0],",",$vpt[1],",",$vpt[2],",",
$vpr[0],",",$vpr[1],",",$vpr[2],",",
$vpd,
"\n"));



