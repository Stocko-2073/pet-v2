include <BOSL2/std.scad>
include <MCAD/servos.scad>
include <globals.scad>
include <BOSL2/screws.scad>

include <roboscad.scad>

module inr14500() {
    %up(25) union() {
        color("#748d") cyl(d=14,h=50-ep, chamfer=0.5);
        color("#888d") cyl(d=11,h=50);
        color("#888d") up(25) cyl(d=5,h=2,anchor=BOT,chamfer2=0.5);
    }
}

module battery_tab_vcc() {
    %color("#888d") render() {
        difference() {
            union() {
                cuboid([12,12,0.25],rounding=1,edges="Z",anchor=BOT);
                left(6) cuboid([7,3,0.25],rounding=1.5,edges=[LEFT+FWD,LEFT+BACK],anchor=BOT+RIGHT);
                up(0.25) {
                    cyl(d1=7,d2=4,h=1.6,rounding1=-1,rounding2=1,anchor=BOT);
                    yflip_copy() fwd(5.75) cuboid([2.5,1,0.75],chamfer=0.75,edges=[LEFT+TOP,RIGHT+TOP],anchor=BOT+FWD);
                }
            }
            down(ep) {
                cyl(d1=7,d2=4,h=1.6,rounding1=-1,rounding2=1,anchor=BOT);
                yflip_copy() fwd(5.75) cuboid([2.5,1,0.75],chamfer=0.75,edges=[LEFT+TOP,RIGHT+TOP],anchor=BOT+FWD);
                left(6+7-1.5) cyl(d=1.25,h=1);
            }
        }
    }
}

module battery_tab_gnd() {
    %color("#888d") render() {
        difference() {
            union() {
                cuboid([12,12,0.25],rounding=1,edges="Z",anchor=BOT);
                left(6) cuboid([7,3,0.25],rounding=1.5,edges=[LEFT+FWD,LEFT+BACK],anchor=BOT+RIGHT);
                up(0.25) {
                    cyl(d1=7,d2=6,h=4,rounding=0.25, anchor=BOT);
                    yflip_copy() fwd(5.75) cuboid([2.5,1,0.75],chamfer=0.75,edges=[LEFT+TOP,RIGHT+TOP],anchor=BOT+FWD);
                }
            }
            down(ep) {
                up(0.25+ep) {
                    cyl(d1=6,d2=5,h=5,rounding=-0.25, anchor=BOT);
                }
                cyl(d=2.5,h=1,anchor=BOT);
                yflip_copy() fwd(5.75) cuboid([2.5,1,0.75],chamfer=0.75,edges=[LEFT+TOP,RIGHT+TOP],anchor=BOT+FWD);
                left(6+7-1.5) cyl(d=1.25,h=1);
            }
        }
    }
}

module battery_tab_neg(len=2) {
    cuboid([12,12,1+$slop],anchor=BOT);
    cyl(d=7,h=6,anchor=BOT);
    cuboid([len+6,7,6],anchor=BOT+LEFT);
    left(6-ep) cuboid([len+ep,3,1+$slop],anchor=BOT+RIGHT);
    right(6-ep) cuboid([len+ep,12,1+$slop],anchor=BOT+LEFT);
}
*!union() {
    battery_tab_vcc();
    battery_tab_gnd();
    #%battery_tab_neg();
}

module 2s_14500_hardware() {
    wall=2;
    x=58;
    up(wall+7) {
        left(x/2) {
            back(15/2) xrot(180) yrot(90) battery_tab_vcc();
            fwd(15/2) xrot(180) yrot(90) battery_tab_gnd();
        }
        zrot(180) left(x/2) {
            back(15/2) xrot(180) yrot(90) battery_tab_vcc();
            fwd(15/2) xrot(180) yrot(90) battery_tab_gnd();
        }
        left(53/2-1.5) fwd(15/2) yrot(90) inr14500(); // battery 1
        right(52/2-1) back(15/2) yrot(-90) inr14500(); // battery 2
    
        %color("#388d") cuboid([41,9,2], anchor=BOT); // BMS
    }
}

module 2s_14500() {
    wall=2;
    x=58;
    difference() {
        union() children();
        up(wall) {
            cuboid([x-wall*3,15*2,15+ep],anchor=BOT);
        }
        up(wall+7) {
            left(x/2) {
                back(15/2) xrot(180) yrot(90) battery_tab_neg(10);
                fwd(15/2) xrot(180) yrot(90) battery_tab_neg(10);
            }
            zrot(180) left(x/2) {
                back(15/2) xrot(180) yrot(90) battery_tab_neg(10);
                fwd(15/2) xrot(180) yrot(90) battery_tab_neg(10);
            }
        }
    }
}

module protoboard() {
    color("#964") cuboid([70,90,1.6]);
}

module servo_9g() {
    %towerprosg90([0,0,26.7],[180,0,0],0,0,1);
    down(5) children();
}

module servo_9g_top_neg() {
    cyl(d=11.75+$slop*2,h=4+$slop,anchor=TOP);
    left(8.75+$slop) cuboid([(5+$slop*2)*2,4.9+$slop*2,4+$slop],anchor=TOP+LEFT,rounding=4.8/2+$slop,edges="Z");
}

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

module servo_9g_basic_neg($slop=0.2) {
    union() {
        up(4) {
            zrot(180) servo_9g_top_neg();
            left(5.9+$slop) down($slop) cuboid([22.5+$slop*2,11.8+$slop*2,22.7+$slop*2],anchor=BOT+LEFT);
            down($slop) left(5.9+$slop+4.7) up(4.3) cuboid([32.1+$slop*2,11.8+$slop*2,(22.7-4.3)+$slop*2],anchor=BOT+LEFT);
            down(4+$slop) {
                left(8.25) {
                    cyl(d=2,h=10,anchor=BOT);
                    cyl(d=4+$slop*2,h=2,anchor=BOT);
                }
                right(18.9) {
                    cyl(d=2,h=10,anchor=BOT);
                    cyl(d=4+$slop*2,h=2,anchor=BOT);
                }
            }
        }
    }
}

*!nop() {
    #render() servo_9g_basic_neg();
    servo_9g();
}

module hip_mount() {
    wall=2;
    z=30.1;
    cuboid([(127+37+wall)/2,12+wall*2,z+wall],rounding=4+ep,edges=[RIGHT+FWD,RIGHT+BACK],anchor=BOT+LEFT);
}

module body_form_shell(sim=false) {
    r=30;
    h=60;
    wall=2;
    z=30.1;
    union() {
        cyl(d=127,h=h,chamfer1=5,rounding2=r,anchor=BOT);
        hip_mount();
        zrot(120) hip_mount();
        zrot(240) hip_mount();
        if (!sim) zrot_copies(n=3) up(z+wall) right(127/2+6.5) cyl(d1=8,d2=2,h=3,anchor=BOT);
    }
}

module body_form(sim=false) {
    r=30;
    h=60;
    wall=2;
    battery_z=17;
    floor_z=14;
    z=26.9;

    if (!sim) {
        up(battery_z) xrot(180)
        2s_14500()
        xrot(180) down(battery_z);
    }

    intersection() {
        body_form_shell();
        difference() {
            union() {
                if (sim) {
                    body_form_shell(sim);
                } else {
                    difference() {
                        body_form_shell(sim);
                        up(wall+floor_z) cyl(d=127-wall*2,h=h-(wall*2+floor_z),rounding2=r-wall,anchor=BOT);
                        zrot_copies(n=3) right(70) zrot(180) {
                            servo_9g_basic_neg();
                        }
                        battery_cover_neg();
                    }
                }
                fwd(127/2-5) cyl(d=10,h=h,anchor=BOT+FWD);
                zrot(120) fwd(127/2-5) cyl(d=10,h=h,anchor=BOT+FWD);
                zrot(-120) fwd(127/2-5) cyl(d=10,h=h,anchor=BOT+FWD);
            }
            if (!sim) {
                zrot_copies(n=3) fwd(127/2-5) {
                    up(z-wall*4) back(5) {
                        screw_hole("M3,15",head="button",atype="head",anchor=BOT,orient=BOT,counterbore=z);
                        up(wall*5) zrot(90) nut_trap_side(trap_width=12,"M3");
                    }
                }
            }
            if (!sim) {
                zrot_copies(n=3) right(127/2-20) {
                    cuboid([3,12,40]);
                } 
            }
        }
    }
}

module body_sim(color="#ddd") {
    battery_z=17;
    part("body")
    color(color) {
        body_form(sim=true);
        battery_cover();
        accelerometer("body");
    }
    right(70) zrot(90) end_part("body") children(0); // leg 1
    zrot(120) right(70) zrot(90) end_part("body") children(1); // leg 2
    zrot(-120) right(70) zrot(90) end_part("body") children(2); // leg 3
}

module body_bottom(color="#ddd") {
    battery_z=17;
    z=26.9;
    color(color) {
        up(z) zview(true) down(z) body_form();
    }
    children(0); // body_top
    up(z) children(1); // protoboard
    zrot_copies(n=3) right(70) children(2); // leg
    children(3); //battery_cover

    up(battery_z) xrot(180) 2s_14500_hardware();

}

module body_top(color="#f84a") {
    color(color) {
        z=26.9;
        up(z) zview(false) down(z) body_form();
    }
}

module hip_form(sim=false) {
    wall=2;
    wall2=3+wall*3;
    z=30.1+wall2*2+wall;
    z2=30.1+wall+$slop*4;
    x=25+4+wall*2+5.9;
    y=26.5+wall;
    render() difference() {
        union() {
            right(x/2-4) down(wall2) union() {
                cuboid([x/2+wall,y,z],rounding=y/2,edges=[LEFT+FWD,LEFT+BACK],anchor=BOT+RIGHT);
                cuboid([x/2+wall*2,y,z],rounding=x/2,edges=[RIGHT+TOP,RIGHT+BOT],anchor=BOT+LEFT);
            }
            if (!sim) {
                back(y) fwd(13.375+wall/2) up(10.5) right(x/2+wall*2+wall/2) yrot(180) xrot(-90) zrot(90)
                cyl(d1=8,d2=2,h=3,orient=UP,anchor=BOT);
            }
        }
        right(x/2-4) down($slop*2){
            cuboid([x/2+wall,y+ep*2,z2],anchor=BOT+RIGHT);
        }
        if (!sim) fwd(7) right(18) {
            down(wall2/2) {
                screw_hole("M3,15",head="button",atype="head",anchor=BOT,orient=FWD,counterbore=x);
                back(11) nut_trap_inline(12,"M3",orient=BACK);
            }
            up(z-wall2*1.5) {
                screw_hole("M3,15",head="button",atype="head",anchor=BOT,orient=FWD,counterbore=x);
                back(11) nut_trap_inline(12,"M3",orient=BACK);
            }
        }
        if (!sim) fwd(13.375+wall/2) up(10.5) right(x/2+wall*2+wall/2) yrot(180) xrot(-90) zrot(90) servo_9g_neg();
        if (!sim) up(z2-$slop*2-ep) cyl(d1=8,d2=2,h=3,orient=UP,anchor=BOT);
        if (!sim) up(ep) {
            servo9g_gear_socket();
            cyl(d=2,h=10,anchor=TOP);
            down(3+2) cyl(d=6.5,h=10,anchor=TOP);
        }
        if (!sim) down(wall2) right(25) cuboid([3,12.2,10],anchor=BOT);
    }
}

module hip_bottom(color="#777") {
    wall=2;
    x=25+4+wall*2+5.9;
    color(color) {
        y=5.075+wall/2;
        fwd(y) yview() back(y)
        hip_form();
    }
    children(0); // hip_top
    zrot(180) children(1); // hip servo 1
    fwd(13.375+wall/2) up(10.5) right(x/2+wall*2+wall/2) yrot(180) xrot(-90) zrot(90) children(2); // hip servo 2
}

module hip_top(color="#777") {
    wall=2;
    color(color) {
        y=5.075+$slop+wall/2;
        fwd(y) yview(true) back(y)
        hip_form();
    }
}

module hip_sim(color="#777") {
    wall=2;
    x=25+4+wall*2+5.9;
    zrot(-90) {
        color(color) hip_form(sim=true);
        fwd(13.375+wall/2) up(10.5) right(x/2+wall*2+wall/2) yrot(180) xrot(-90) zrot(90) 
        zrot(180+45)
        children(0); // leg_sim
    }
}

module battery_cover() {
    wall=2;
    x=58;
    d=7;
    color("#bbb") difference() {
        union() {
            cuboid([x+wall*2,15*2+wall*4,wall],chamfer=wall,edges=BOT,anchor=TOP);
            cuboid([d,15+wall*2+d,wall],rounding=d/2,edges="Z",anchor=TOP+FWD);
            fwd(15+wall*2) cuboid([30,wall*2,wall],chamfer=wall,edges=[BOT+FWD,TOP+BACK],anchor=BOT);
        }
        back(15+wall*2+d/2) down(wall) yrot(-90) screw_hole("M3,6",head="button",atype="head",anchor=BOT,orient=LEFT);
    }
}

module battery_cover_neg() {
    wall=2;
    x=58;
    d=7;
    fwd(15+wall*2) cuboid([30+$slop*2,wall*2+$slop*2,wall+$slop*2],chamfer=wall+$slop,edges=[BOT+FWD,TOP+BACK],anchor=BOT);
    back(15+wall*2+d/2) down(wall) yrot(-90) screw_hole("M3,6",head="button",atype="head",anchor=BOT,orient=LEFT);
    back(15+wall*2+d/2) up(wall) zrot(-90) nut_trap_side(trap_width=12,"M3");
}

module leg_form(sim=false) {
    wall=2;
    wall2=3;
    z=26.5+wall;
    z2=z+wall2*2+wall*2;
    h=53;
    h2=53+4;
    y=23.5;
    
    difference() {
        union() {
            back(2.75) up(z2/2) {
                hull() {
                    cyl(d=y,h=z2, chamfer=2);
                    left(h2/2) cyl(d=y,h=z2, chamfer=2);
                }
                hull() {
                    left(h2/2) cyl(d=y,h=z);
                    fwd((y-16)/2) left(h+8) {
                        cuboid([12,16,z],rounding=4,edges="Z");
                    }
                }
            }
            if (!sim) fwd((y-16)/2) back(2.75) up(y+10) left(h)
                cyl(d1=8,d2=2,h=3,anchor=BOT);
        }
        if (!sim) back(2.75) left(h2/2) up(13.3-$slop-wall) {
            screw_hole("M3,15",head="button",atype="head",anchor=BOT,orient=BOT,counterbore=z);
            up(wall*5) zrot(90) nut_trap_inline(30,"M3");
        }
        back(2.75) up(z2/2) difference() {
            x=h2/2+6;
            cuboid([x*2,y+ep*2,z+$slop*4]);
            left(x) cyl(d=y+ep*2,h=z+$slop*4);
        }
        if (!sim) fwd((y-16)/2) back(2.75) up(5) left(h) servo_9g_neg();
        if (!sim) up(y+10) cyl(d1=8,d2=2,h=3,anchor=BOT);
        if (!sim) up(5) {
            servo9g_gear_socket();
            cyl(d=2,h=10,anchor=TOP);
            down(3+2) cyl(d=6.5,h=10,anchor=TOP);
        }
    }
}

module leg_bottom(color="#ddd") {
    wall=2;
    h=48+5;
    y=23.5;
    z=13.3;
    render() color(color) 
    up(0.1) up(z)
    zview() 
    down(z)
    leg_form();
    children(0); // leg_top
    fwd((y-16)/2) back(2.75) up(5) left(h) children(1); // foot servo
}

module leg_top(color="#ddd") {
    z=13.3-$slop;
    render() color(color) 
    up(z)
    zview(true) 
    down(z) 
    leg_form();
}

module leg_sim(color="#ddd") {
    wall=2;
    h=48+5;
    y=23.5;
    zrot(90) {
        render() color(color) 
        up(0.1) down(5) 
        leg_form(sim=true);
        fwd((y-16)/2) back(2.75) up(5) left(h) 
        zrot(-90)
        children(0); // foot_sim
    }
}

module foot_form(sim=false) {
    wall=2;
    wall2=3;
    z=26.5+wall;
    z2=z+wall2*2+wall*2;
    h=53;
    h2=53+4;
    y=23.5;
    
    difference() {
        union() {
            up(z2/2) {
                hull() {
                    cyl(d=y,h=z2, chamfer=2);
                    left(h2/2) cyl(d=y,h=z2, chamfer=2);
                }
                left(h2/2) hull() {
                    cyl(d=y,h=z);
                    left(8) sphere(d=y);
                }
            }
        }
        if (!sim) left(h2/2) up(z2/2-wall*2) {
            screw_hole("M3,15",head="button",atype="head",anchor=BOT,orient=BOT,counterbore=z);
            up(wall*5) zrot(90) nut_trap_inline(30,"M3");
        }
        up(z2/2) difference() {
            x=h2/2-6;
            cuboid([x*2,y+ep*2,z+$slop*4],rounding=4,edges=[LEFT+FWD,LEFT+BACK]);
            //left(x) cyl(d=y+ep*2,h=z+$slop*4);
        }
        if (!sim) up(y+10) cyl(d1=8,d2=2,h=3,anchor=BOT);
        if (!sim) up(5) {
            servo9g_gear_socket();
            cyl(d=2,h=10,anchor=TOP);
            down(3+2) cyl(d=6.5,h=10,anchor=TOP);
        }
    }
}

module foot_bottom(color="#777") {
    wall=2;
    wall2=3;
    z=26.5+wall;
    z2=z+wall2*2+wall*2;
    render() color(color) {
        up(z2/2)
        zview() 
        down(z2/2) 
        foot_form();
    }
    children(0); // foot_top
}

module foot_top(color="#777") {
    wall=2;
    wall2=3;
    z=26.5+wall;
    z2=z+wall2*2+wall*2;
    render() color(color) {
        up(z2/2-$slop)
        zview(true) 
        down(z2/2-$slop) 
        foot_form();
    }
}

module foot_sim(color="#777") {
    wall=2;
    wall2=3;
    z=26.5+wall;
    z2=z+wall2*2+wall*2;
    zrot(90) render() color(color) down(10) foot_form(sim=true);
}

*body_bottom() {
    //up(100) 
    %body_top();
    protoboard();
    hip_bottom() {
        hip_top();
        servo_9g();
        servo_9g() 
            zrot(-30) leg_bottom() {
                leg_top();
                servo_9g() // foot servo
                    zrot(50) foot_bottom() foot_top();
            }
    }
    battery_cover();
}

//left(200) zrot(60)

module whole_leg_sim(i) {
    hip_name=str("hip",i);
    leg_name=str("leg",i);
    foot_name=str("foot",i);
    joint("revolute", "body", hip_name,
        lower_limit=-90, upper_limit=90,
        damping=0.5, friction=0.5, velocity=10, effort=10,
        soft_lower_limit=-90, soft_upper_limit=90)
    part(hip_name) hip_sim() end_part(hip_name)
    joint("revolute", hip_name, leg_name,
        lower_limit=-90, upper_limit=90,
        damping=0.5, friction=0.5, velocity=10, effort=10,
        soft_lower_limit=-90, soft_upper_limit=90)
    part(leg_name) leg_sim() end_part(leg_name)
    joint("revolute", leg_name, foot_name,
        lower_limit=-90, upper_limit=90,
        damping=0.5, friction=0.5, velocity=10, effort=10,
        soft_lower_limit=-90, soft_upper_limit=90)
    part(foot_name) foot_sim() end_part(foot_name);
}

$fs=4;
                    
body_sim() {
    whole_leg_sim(1);
    whole_leg_sim(2);
    whole_leg_sim(3);
}