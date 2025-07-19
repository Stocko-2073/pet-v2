use <override.scad>

////////////////////////
// URDF EXPORT SUPPORT
////////////////////////
$part="";
show_joints = true;
joint_radius = 45;
ep = is_undef(ep) ? 0.02 : ep;

module _translate(v) {
    if (!is_undef(roboscad)) echo("roboscad","translate",v);
    __translate(v) children();
    if (!is_undef(roboscad)) echo("roboscad","pop");
}

module _rotate(a,v) {
    if (!is_undef(roboscad)) {
        v = is_undef(v) ? [0,0,1] : v;
        if (is_list(a)) {
            echo("roboscad","rotate",a);
        } else if (is_num(a) && is_list(v)) {
            echo("roboscad","rotate",a*v);
        } else {
            echo("roboscad","rotate",a,v);
            assert(false, "Invalid arguments for rotate");
        }
    }
    __rotate(a,v) children();
    if (!is_undef(roboscad)) echo("roboscad","pop");
}

module _scale(v) {
    if (!is_undef(roboscad)) echo("roboscad","scale",v);
    __scale(v) children();
    if (!is_undef(roboscad)) echo("roboscad","pop");
}

module mirror(v) {
    if (!is_undef(roboscad)) echo("roboscad","mirror",v);
    __mirror(v) children();
    if (!is_undef(roboscad)) echo("roboscad","pop");
}

module joint(
    type,
    parent_part,
    child_part,
    lower_limit=-180,
    upper_limit=180,
    damping=0.5,
    friction=0.5,
    velocity=10,
    effort=10,
    soft_lower_limit=-180,
    soft_upper_limit=180,
    axis="0 0 1"
) {
    if (!is_undef(roboscad)) {
        echo("roboscad","joint",type,parent_part,child_part,[lower_limit,upper_limit,damping,friction,velocity,effort,soft_lower_limit,soft_upper_limit,axis]);
    }
    if (show_joints) {
        if (type == "revolute") {
            range = upper_limit - lower_limit;
            zero = -lower_limit;
            %color("#ff80ff80") 
                _rotate(270+lower_limit, [0,0,1])
                down(1) pie_slice(ang=range, l=1-ep, r=joint_radius);
        } else if (type == "fixed") {
            %color("#ff80ff80") 
                cuboid([15,15,1-ep]);
        } else if (type == "bushing") {
            %color("#80ff8080") 
                cyl(r=7.5, h=1-ep);
        }
    }
    down(ep) 
    children();
}

module accelerometer(name) {
    if (!is_undef(roboscad)) {
        echo("roboscad","sensor",name,"accelerometer",$part,[1,1,1]);
    }
    if (show_joints) {
        #%color("#ff8080ff") 
            sphere(d=15);
    }
    children();
}
module touch(name, size=[1,1,1]) {
    if (!is_undef(roboscad)) {
        echo("roboscad","sensor",name,"touch",$part,size);
    }
    if (show_joints) {
        #%color("#ff8080ff") sphere(d=1,$fn=48);
    }
    children();
}

module part(p, additional_mass=0, color="#dd33dd") {
    if (!is_undef(roboscad) && !is_undef(p)) {
        echo("roboscad","start_part",p,$part,additional_mass,color);
        if (p == roboscad) {
            $part = p;
            !children();
        } else {
            $part = p;
            children();
        }
    } else {
        children();
    }
}

module end_part(p) {
    if (!is_undef(roboscad) && !is_undef(p)) {
        if (p == roboscad) {
            echo("roboscad","end_part",p);
        } else {
            children();
        }
    } else {
        children();
    }
}
