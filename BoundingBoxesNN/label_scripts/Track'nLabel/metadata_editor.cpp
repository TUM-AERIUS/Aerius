// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "metadata_editor.h"
#include <dlib/array.h>
#include <dlib/queue.h>
#include <dlib/static_set.h>
#include <dlib/misc_api.h>
#include <dlib/image_io.h>
#include <dlib/array2d.h>
#include <dlib/pixel.h>
#include <sstream>
#include <ctime>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace dlib;

extern const char* VERSION;

rgb_alpha_pixel string_to_color(
    const std::string& str
)
{
    if (str.size() == 0)
    {
        return rgb_alpha_pixel(255,0,0,255);
    }
    else
    {
        // make up a random color based on the string label.
        hsi_pixel pix;
        pix.h = static_cast<unsigned char>(dlib::hash(str)&0xFF); 
        pix.s = 255;
        pix.i = 150;
        rgb_alpha_pixel result;
        assign_pixel(result, pix);
        return result;
    }
}

// ----------------------------------------------------------------------------------------
bool contains_plus(string str){
    cout << __PRETTY_FUNCTION__ << std::endl;
    
    for(int i=0;i<str.size(); ++i){
        if(str[i]=='+')
            return true;
    }
    return false;
}

metadata_editor::
metadata_editor(
    const std::string& filename_
) : 
    mbar(*this),
    lb_images(*this),
    image_pos(0),
    display(*this),
    overlay_label_name(*this),
    overlay_label(*this),
    keyboard_jump_pos(0),
    last_keyboard_jump_pos_update(0)
{
    file metadata_file(filename_);
    filename = metadata_file.full_name();
    // Make our current directory be the one that contains the metadata file.  We 
    // do this because that file might contain relative paths to the image files
    // we are supposed to be loading.
    set_current_dir(get_parent_directory(metadata_file).full_name());

    load_image_dataset_metadata(metadata, filename);

    dlib::array<std::string>::expand_1a files;
    files.resize(metadata.images.size());
    for (unsigned long i = 0; i < metadata.images.size(); ++i)
    {
        files[i] = metadata.images[i].filename;
    }
    lb_images.load(files);
    lb_images.enable_multiple_select();

    lb_images.set_click_handler(*this, &metadata_editor::on_lb_images_clicked);

    overlay_label_name.set_text("Next Label: ");
    overlay_label.set_width(200);

    display.set_overlay_rects_changed_handler(*this, &metadata_editor::on_overlay_rects_changed);
    display.set_overlay_rect_selected_handler(*this, &metadata_editor::on_overlay_rect_selected);
    overlay_label.set_text_modified_handler(*this, &metadata_editor::on_overlay_label_changed);

    mbar.set_number_of_menus(2);
    mbar.set_menu_name(0,"File",'F');
    mbar.set_menu_name(1,"Help",'H');


    mbar.menu(0).add_menu_item(menu_item_text("Save",*this,&metadata_editor::file_save,'S'));
    mbar.menu(0).add_menu_item(menu_item_text("Save As",*this,&metadata_editor::file_save_as,'A'));
    mbar.menu(0).add_menu_item(menu_item_separator());
    mbar.menu(0).add_menu_item(menu_item_text("Remove Selected Images",*this,&metadata_editor::remove_selected_images,'R'));
    mbar.menu(0).add_menu_item(menu_item_separator());
    mbar.menu(0).add_menu_item(menu_item_text("Exit",static_cast<base_window&>(*this),&drawable_window::close_window,'x'));

    mbar.menu(1).add_menu_item(menu_item_text("About",*this,&metadata_editor::display_about,'A'));

    // set the size of this window.
    on_window_resized();
    load_image_and_set_size(0);
    on_window_resized();
    if (image_pos < lb_images.size() )
        lb_images.select(image_pos);

    // make sure the window is centered on the screen.
    unsigned long width, height;
    get_size(width, height);
    unsigned long screen_width, screen_height;
    get_display_size(screen_width, screen_height);
    set_pos((screen_width-width)/2, (screen_height-height)/2);

    correspondance = std::vector<std::vector<std::pair<int,int> > >(metadata.images.size());
    box_ids = 0;
    show();
    
    for(int c = 0; c < metadata.images.size(); ++c){
        for(int k = 0 ; k < metadata.images[c].boxes.size();++k){
            if(contains_plus(metadata.images[c].boxes[k].label)){
               
                int label_nr_id = 0;
                for(int an = metadata.images[c].boxes[k].label.size()-5; an<=metadata.images[c].boxes[k].label.size()-1;++an){
                    char chr =metadata.images[c].boxes[k].label[an];
                    int nr = chr - 48;
                    label_nr_id*=10;
                    label_nr_id+=nr;
                }
                
                if(box_ids<=label_nr_id){
                    box_ids=label_nr_id+1;
                }
            }
        }

    }
}

// ----------------------------------------------------------------------------------------

metadata_editor::
~metadata_editor(
)
{
    close_window();
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
add_labelable_part_name (
    const std::string& name
)
{
    cout<<"add_labelable_part_name"<<endl;
    display.add_labelable_part_name(name);
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
file_save()
{
    cout<<"file_save"<<endl;
    save_metadata_to_file(filename);
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
save_metadata_to_file (
    const std::string& file
)
{
    cout << __PRETTY_FUNCTION__ << std::endl;

    cout<<"save_metadata_to_file"<<endl;
    try
    {
        save_image_dataset_metadata(metadata, file);
    }
    catch (dlib::error& e)
    {
        message_box("Error saving file", e.what());
    }
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
file_save_as()
{
    cout<<"file_save_as"<<endl;
    save_file_box(*this, &metadata_editor::save_metadata_to_file);
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
remove_selected_images()
{
    cout << __PRETTY_FUNCTION__ << std::endl;

    cout<<"remove_selected_images"<<endl;
    dlib::queue<unsigned long>::kernel_1a list;
    lb_images.get_selected(list);
    list.reset();
    unsigned long min_idx = lb_images.size();
    while (list.move_next())
    {
        lb_images.unselect(list.element());
        min_idx = std::min(min_idx, list.element());
    }


    // remove all the selected items from metadata.images
    dlib::static_set<unsigned long>::kernel_1a to_remove;
    to_remove.load(list);
    std::vector<dlib::image_dataset_metadata::image> images;
    for (unsigned long i = 0; i < metadata.images.size(); ++i)
    {
        if (to_remove.is_member(i) == false)
        {
            images.push_back(metadata.images[i]);
        }
    }
    images.swap(metadata.images);


    // reload metadata into lb_images
    dlib::array<std::string>::expand_1a files;
    files.resize(metadata.images.size());
    for (unsigned long i = 0; i < metadata.images.size(); ++i)
    {
        files[i] = metadata.images[i].filename;
    }
    lb_images.load(files);


    if (min_idx != 0)
        min_idx--;
    select_image(min_idx);
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
on_window_resized(
)
{
    cout << __PRETTY_FUNCTION__ << std::endl;

    cout<<"on_window_resized"<<endl;
    drawable_window::on_window_resized();

    unsigned long width, height;
    get_size(width, height);

    lb_images.set_pos(0,mbar.bottom()+1);
    lb_images.set_size(180, height - mbar.height());

    overlay_label_name.set_pos(lb_images.right()+10, mbar.bottom() + (overlay_label.height()-overlay_label_name.height())/2+1);
    overlay_label.set_pos(overlay_label_name.right(), mbar.bottom()+1);
    display.set_pos(lb_images.right(), overlay_label.bottom()+3);

    display.set_size(width - display.left(), height - display.top());
}

// ----------------------------------------------------------------------------------------

void propagate_labels(
    const std::string& label,
    dlib::image_dataset_metadata::dataset& data,
    unsigned long prev,
    unsigned long next
)
{
    cout << __PRETTY_FUNCTION__ << std::endl;

    cout<<"propagate_labels"<<endl;
    if (prev == next || next >= data.images.size())
        return;


    for (unsigned long i = 0; i < data.images[prev].boxes.size(); ++i)
    {
        if (data.images[prev].boxes[i].label != label)
            continue;

        // figure out which box in the next image matches the current one the best
        const rectangle cur = data.images[prev].boxes[i].rect;
        double best_overlap = 0;
        unsigned long best_idx = 0;
        for (unsigned long j = 0; j < data.images[next].boxes.size(); ++j)
        {
            const rectangle next_box = data.images[next].boxes[j].rect;
            const double overlap = cur.intersect(next_box).area()/(double)(cur+next_box).area();
            if (overlap > best_overlap)
            {
                best_overlap = overlap;
                best_idx = j;
            }
        }

        // If we found a matching rectangle in the next image and the best match doesn't
        // already have a label.
        if (best_overlap > 0.5 && data.images[next].boxes[best_idx].label == "")
        {
            data.images[next].boxes[best_idx].label = label;
        }
    }

}
// ----------------------------------------------------------------------------------------
//void metadata_editor::on_mouse_move (unsigned long state,long x,long y){
//    lastx=x;
//    lasty=y;
//}

//void metadata_editor::on_mouse_up (
//                  unsigned long btn,
//                  unsigned long state,
//                  long x,
//                  long y
//                                   ){
//    lastx = x;
//    lasty = y;
//}

// ----------------------------------------------------------------------------------------
bool begins_with(string str){
    return (str[0] == 't' && str[1] == '_') ;
}
void metadata_editor::propagate_labels2(
                      dlib::image_dataset_metadata::dataset& data,
                      unsigned long prev,
                      unsigned long next
                      )
{
    cout << __PRETTY_FUNCTION__ << std::endl;

    cout<<"propagate_labels2"<<endl;
    if (prev == next || next >= data.images.size() || image_pos >= data.images.size()-1)
        return;
    
    //cout<<"how many boxes: "<<data.images[image_pos].boxes.size()<<endl;
    load_image(image_pos);
    for (unsigned long i = 0; i < data.images[image_pos].boxes.size(); ++i)
    {
        
        bool is_new = true;
        bool track = true;
        if(!begins_with(data.images[image_pos].boxes[i].label)){
            track = false;
            continue;
        }
        int corresponding_tracker;
        std::map<std::string,int>::iterator it = map_id_to_index.find(data.images[image_pos].boxes[i].label);
        if(it != map_id_to_index.end())
        {
            
            corresponding_tracker = it->second;
            is_new = false;
            cout<<"c_tracker: "<<corresponding_tracker<<endl;
        }
        
        if(!is_new){
            cout<<"entered not new "<<endl;
            array2d<rgb_pixel> img_next;
            try
            {
                dlib::load_image(img_next, data.images[image_pos+1].filename);
            }
            catch (exception& e)
            {
                message_box("Error loading image", e.what());
            }
            
            trackers[corresponding_tracker].update(img_next);
            dlib::drectangle found_position = trackers[corresponding_tracker].get_position();
            bool insert = true;
            //cout<<"seg"<<endl;
            
            for(int k = 0 ; k < data.images[image_pos+1].boxes.size();++k){
                if(data.images[image_pos+1].boxes[k].label == data.images[image_pos].boxes[i].label){
                    insert = false;
                    break;
                }
            }
            
            if(insert){
                dlib::image_dataset_metadata::box b(found_position);
                b.label = data.images[image_pos].boxes[i].label;
                data.images[image_pos+1].boxes.push_back(b);
                correspondance[image_pos+1].push_back(std::make_pair(data.images[image_pos+1].boxes.size()-1,corresponding_tracker));
                //map_id_to_index[b.label] = trackers.size()-1;
                //map_id_to_box[b.label].push_back(make_pair(image_pos+1,data.images[image_pos+1].boxes.size()-1));
                box_position_to_id[make_pair(image_pos+1,data.images[image_pos+1].boxes.size()-1)] = b.label;
            }
            
        }else{
            cout<<"entered new "<<endl;
            dlib::correlation_tracker tracker = *new dlib::correlation_tracker;
            trackers.push_back(tracker);
            const rectangle cur = data.images[image_pos].boxes[i].rect;
            double midx = (cur.left() + cur.right() )/2;
            double midy = (cur.top() + cur.bottom()) /2;
            
            double length = std::abs(cur.left() - cur.right() );
            double height = std::abs(cur.top() - cur.bottom() );
            
            //length+=length*0.02;
            //height+=height*0.02;
            
            cout<<length<<endl<<height<<endl;
            
            
            
            array2d<rgb_pixel> img,img_next;
            try
            {
                dlib::load_image(img, data.images[image_pos].filename);
            }
            catch (exception& e)
            {
                message_box("Error loading image", e.what());
            }
            
            trackers[trackers.size()-1].start_track(img, centered_rect(point(midx,midy), length, height));
            
            try
            {
                dlib::load_image(img_next, data.images[image_pos+1].filename);
            }
            catch (exception& e)
            {
                message_box("Error loading image", e.what());
            }
            
            trackers[trackers.size()-1].update(img_next);
            dlib::drectangle found_position = trackers[trackers.size()-1].get_position();
            cout<<"box sizes tracker: "<<found_position.left()<<" "<<found_position.top()<<" "<<found_position.right()<<" "<<found_position.bottom()<<endl;

            
            dlib::image_dataset_metadata::box b(found_position);
            b.label = data.images[image_pos].boxes[i].label;
            data.images[image_pos+1].boxes.push_back(b);
            map_id_to_index[b.label] = trackers.size()-1;
            correspondance[image_pos+1].push_back( std::make_pair(data.images[image_pos+1].boxes.size()-1,trackers.size()-1));
            //map_id_to_box[b.label].push_back(make_pair(image_pos+1,data.images[image_pos+1].boxes.size()-1));
            box_position_to_id[make_pair(image_pos+1,data.images[image_pos+1].boxes.size()-1)] = b.label;
            //cout<<"from: "<<image_pos<<" to: "<<image_pos+1<<endl;
            //cout<<"pushed: ("<<data.images[image_pos+1].boxes.size()-1<<","<<trackers.size()-1<<")"<<endl;
        }
        
        // If we found a matching rectangle in the next image and the best match doesn't
        // already have a label.
        }
    
}


// ----------------------------------------------------------------------------------------

bool has_label_or_all_boxes_labeled (
    const std::string& label,
    const dlib::image_dataset_metadata::image& img 
)
{
    cout << __PRETTY_FUNCTION__ << std::endl;

    //cout<<"has_label_or_all_boxes_labeled"<<endl;
    if (label.size() == 0)
        return true;

    bool all_boxes_labeled = true;
    for (unsigned long i = 0; i < img.boxes.size(); ++i)
    {
        if (img.boxes[i].label == label)
            return true;
        if (img.boxes[i].label.size() == 0)
            all_boxes_labeled = false;
    }

    return all_boxes_labeled;
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
on_keydown (
    unsigned long key,
    bool is_printable,
    unsigned long state
)
{
    cout << __PRETTY_FUNCTION__ << std::endl;

    //cout<<"on_keydown"<<endl;
    drawable_window::on_keydown(key, is_printable, state);

    if (is_printable)
    {
        if (key == '\t')
        {
            overlay_label.give_input_focus();
            overlay_label.select_all_text();
        }

        // If the user types a number then jump to that image.
        if ('0' <= key && key <= '9' && metadata.images.size() != 0 && !overlay_label.has_input_focus())
        {
            time_t curtime = time(0);
            // If it's been a while since the user typed numbers then forget the last jump
            // position and start accumulating numbers over again.
            if (curtime-last_keyboard_jump_pos_update >= 2)
                keyboard_jump_pos = 0;
            last_keyboard_jump_pos_update = curtime;

            keyboard_jump_pos *= 10;
            keyboard_jump_pos += key-'0';
            if (keyboard_jump_pos >= metadata.images.size())
                keyboard_jump_pos = metadata.images.size()-1;

            image_pos = keyboard_jump_pos;
            select_image(image_pos);
        }
        else
        {
            if(key == 65 || key == 97){
                std::cout<<"pressed a or A: "<<std::endl;
            }
            last_keyboard_jump_pos_update = 0;
        }

        return;
    }

    if (key == base_window::KEY_UP)
    {
        if (state&base_window::KBD_MOD_CONTROL)
        {
            // If the label we are supposed to propagate doesn't exist in the current image
            // then don't advance.
            if (!has_label_or_all_boxes_labeled(display.get_default_overlay_rect_label(),metadata.images[image_pos]))
                return;

            // if the next image is going to be empty then fast forward to the next one
            while (image_pos > 1 && metadata.images[image_pos-1].boxes.size() == 0)
                --image_pos;

            propagate_labels(display.get_default_overlay_rect_label(), metadata, image_pos, image_pos-1);
        }else{
            //propagate_labels2(metadata, image_pos, image_pos-1);
        }
        select_image(image_pos-1);
    }
    else if (key == base_window::KEY_DOWN)
    {
        if (state&base_window::KBD_MOD_CONTROL)
        {
            // If the label we are supposed to propagate doesn't exist in the current image
            // then don't advance.
            if (!has_label_or_all_boxes_labeled(display.get_default_overlay_rect_label(),metadata.images[image_pos]))
                return;

            // if the next image is going to be empty then fast forward to the next one
            while (image_pos+1 < metadata.images.size() && metadata.images[image_pos+1].boxes.size() == 0)
                ++image_pos;

            propagate_labels(display.get_default_overlay_rect_label(), metadata, image_pos, image_pos+1);
        }else{
            propagate_labels2(metadata, image_pos, image_pos+1);
        }
        select_image(image_pos+1);
    }
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
select_image(
    unsigned long idx
)
{
    cout << __PRETTY_FUNCTION__ << std::endl;

    //cout<<"select_image"<<endl;
    if (idx < lb_images.size())
    {
        // unselect all currently selected images
        dlib::queue<unsigned long>::kernel_1a list;
        lb_images.get_selected(list);
        list.reset();
        while (list.move_next())
        {
            lb_images.unselect(list.element());
        }


        lb_images.select(idx);
        load_image(idx);
    }
    else if (lb_images.size() == 0)
    {
        display.clear_overlay();
        array2d<unsigned char> empty_img;
        display.set_image(empty_img);
    }
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
on_lb_images_clicked(
    unsigned long idx
) 
{
    cout << __PRETTY_FUNCTION__ << std::endl;

    //cout<<"on_lb_images_clicked"<<endl;
    load_image(idx);
}

// ----------------------------------------------------------------------------------------

std::vector<dlib::image_display::overlay_rect> get_overlays (
    const dlib::image_dataset_metadata::image& data
)
{
    cout << __PRETTY_FUNCTION__ << std::endl;

    //cout<<"get_overlays"<<endl;
    std::vector<dlib::image_display::overlay_rect> temp(data.boxes.size());
    for (unsigned long i = 0; i < temp.size(); ++i)
    {
        temp[i].rect = data.boxes[i].rect;
        temp[i].label = data.boxes[i].label;
        temp[i].parts = data.boxes[i].parts;
        temp[i].crossed_out = data.boxes[i].ignore;
        temp[i].color = string_to_color(data.boxes[i].label);
    }
    return temp;
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
load_image(
    unsigned long idx
)
{
    cout << __PRETTY_FUNCTION__ << std::endl;

    //cout<<"load_image"<<endl;
    if (idx >= metadata.images.size())
        return;
    cout<<"load image"<<endl;
    image_pos = idx; 

    array2d<rgb_pixel> img;
    display.clear_overlay();
    try
    {
        dlib::load_image(img, metadata.images[idx].filename);
        set_title(metadata.name + ": " +metadata.images[idx].filename);
    }
    catch (exception& e)
    {
        message_box("Error loading image", e.what());
    }

    display.set_image(img);
    display.add_overlay(get_overlays(metadata.images[idx]));
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
load_image_and_set_size(
    unsigned long idx
)
{
    cout << __PRETTY_FUNCTION__ << std::endl;

    //cout<<"load_image_and_set_size"<<endl;
    if (idx >= metadata.images.size())
        return;

    image_pos = idx; 

    array2d<rgb_pixel> img;
    display.clear_overlay();
    try
    {
        dlib::load_image(img, metadata.images[idx].filename);
        set_title(metadata.name + ": " +metadata.images[idx].filename);
    }
    catch (exception& e)
    {
        message_box("Error loading image", e.what());
    }


    unsigned long screen_width, screen_height;
    get_display_size(screen_width, screen_height);


    unsigned long needed_width = display.left() + img.nc() + 4;
    unsigned long needed_height = display.top() + img.nr() + 4;
	if (needed_width < 300) needed_width = 300;
	if (needed_height < 300) needed_height = 300;

    if (needed_width > 100 + screen_width)
        needed_width = screen_width - 100;
    if (needed_height > 100 + screen_height)
        needed_height = screen_height - 100;

    set_size(needed_width, needed_height);


    display.set_image(img);
    display.add_overlay(get_overlays(metadata.images[idx]));
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
on_overlay_rects_changed(
)
{
    cout << __PRETTY_FUNCTION__ << std::endl;
    //cout<<"on_overlay_rects_changed"<<endl;
    using namespace dlib::image_dataset_metadata;
    
    if (image_pos < metadata.images.size())
    {
        string new_label = trim(overlay_label.text());
        if(!contains_plus(new_label)){
            box_ids++;
            string string_id = std::to_string(box_ids);
            new_label = new_label.append("+");
            string nr = std::string(5-string_id.size(),'0').append(string_id);
            new_label+=nr;
            
        }
        cout<<"on_overlay_label_changed"<<endl;
        display.set_default_overlay_rect_label(new_label);
        display.set_default_overlay_rect_color(string_to_color(new_label));
        
        const std::vector<image_display::overlay_rect>& rects = display.get_overlay_rects();
        
        cout << "number of overlay rects: "<< rects.size() << endl;

        std::vector<box>& boxes = metadata.images[image_pos].boxes;
        
        boxes.clear();
        
        for (unsigned long i = 0; i < rects.size(); ++i)
        {
            box temp;
            std::map<std::string,int>::iterator it = map_id_to_index.find(rects[i].label);
            if(it != map_id_to_index.end())
            {
                int corresponding_tracker = it->second;
                array2d<rgb_pixel> img2;
                dlib::load_image(img2, metadata.images[image_pos].filename);
                const rectangle cur = metadata.images[image_pos].boxes[i].rect;
                double midx = (cur.left() + cur.right() )/2;
                double midy = (cur.top() + cur.bottom()) /2;
                
                double length = std::abs(cur.left() - cur.right() );
                double height = std::abs(cur.top() - cur.bottom() );
                drectangle tracker_rect = trackers[corresponding_tracker].get_position();
                
                if(std::abs(tracker_rect.left() - rects[i].rect.left()) >=2 ||  std::abs(tracker_rect.right() - rects[i].rect.right()) >=2 || std::abs(tracker_rect.top() - rects[i].rect.top()) >=2 || std::abs(tracker_rect.bottom() - rects[i].rect.bottom()) >=2){
                    trackers[corresponding_tracker].start_track(img2, centered_rect(point(midx,midy), length, height));
                }
                
            }
            
            temp.label = rects[i].label;
            temp.rect = rects[i].rect;
            temp.parts = rects[i].parts;
            temp.ignore = rects[i].crossed_out;
            boxes.push_back(temp);
        }
        
        
        
    }
}

// ----------------------------------------------------------------------------------------
void metadata_editor::
on_overlay_label_changed(
)
{
    cout << __PRETTY_FUNCTION__ << std::endl;

    string new_label = trim(overlay_label.text());
    if(!contains_plus(new_label)){
        cout<<"enter here"<<endl;
        string string_id = std::to_string(box_ids);
        new_label = new_label.append("+");
        string nr = std::string(5-string_id.size(),'0').append(string_id);
        new_label+=nr;
        //box_ids++;
    }
    cout<<"on_overlay_label_changed"<<endl;
    display.set_default_overlay_rect_label(new_label);
    display.set_default_overlay_rect_color(string_to_color(new_label));
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
on_overlay_rect_selected(
    const image_display::overlay_rect& orect
)
{
    cout << __PRETTY_FUNCTION__ << std::endl;

    //cout<<"on_overlay_rect_selected"<<endl;
    overlay_label.set_text(orect.label);
    display.set_default_overlay_rect_label(orect.label);
    display.set_default_overlay_rect_color(string_to_color(orect.label));
    cout<<"plm1"<<endl;
}

// ----------------------------------------------------------------------------------------

void metadata_editor::
display_about(
)
{
    cout << __PRETTY_FUNCTION__ << std::endl;

    //cout<<"display_about"<<endl;
    std::ostringstream sout;
    sout << wrap_string("Image Labeler v" + string(VERSION) + "." ,0,0) << endl << endl;
    sout << wrap_string("This program is a tool for labeling images with rectangles. " ,0,0) << endl << endl;

    sout << wrap_string("You can add a new rectangle by holding the shift key, left clicking "
                        "the mouse, and dragging it.  New rectangles are given the label from the \"Next Label\" "
                        "field at the top of the application.  You can quickly edit the contents of the Next Label field "
                        "by hitting the tab key. Double clicking "
                        "a rectangle selects it and the delete key removes it.  You can also mark "
                        "a rectangle as ignored by hitting the i key when it is selected.  Ignored "
                        "rectangles are visually displayed with an X through them."
                        ,0,0) << endl << endl;

    sout << wrap_string("It is also possible to label object parts by selecting a rectangle and "
                        "then right clicking.  A popup menu will appear and you can select a part label. "
                        "Note that you must define the allowable part labels by giving --parts on the "
                        "command line.  An example would be '--parts \"leye reye nose mouth\"'."
                        ,0,0) << endl << endl;

    sout << wrap_string("Additionally, you can hold ctrl and then scroll the mouse wheel to zoom.  A normal left click "
                        "and drag allows you to navigate around the image.  Holding ctrl and "
                        "left clicking a rectangle will give it the label from the Next Label field. "
                        "Holding shift + right click and then dragging allows you to move things around. "
                        "Holding ctrl and pressing the up or down keyboard keys will propagate "
                        "rectangle labels from one image to the next and also skip empty images. " 
                        "Finally, typing a number on the keyboard will jump you to a specific image.",0,0) << endl;

    message_box("About Image Labeler",sout.str());
}

// ----------------------------------------------------------------------------------------

