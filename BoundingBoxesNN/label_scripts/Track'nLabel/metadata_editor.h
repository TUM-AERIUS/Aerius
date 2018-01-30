// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_METADATA_EdITOR_H__
#define DLIB_METADATA_EdITOR_H__

#include <dlib/gui_widgets.h>
#include "dlib/data_io.h"
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/dir_nav.h>
#include <vector>
#include <map>
// ----------------------------------------------------------------------------------------

class metadata_editor : public dlib::drawable_window 
{
public:
    metadata_editor(
        const std::string& filename_
    );

    ~metadata_editor();

    void add_labelable_part_name (
        const std::string& name
    );
    unsigned long image_pos;
    std::vector<dlib::correlation_tracker> trackers;
    std::vector<std::vector<std::pair<int,int> > > correspondance;
    std::map<std::pair<int,int>, std::string> box_position_to_id;
    std::vector<std::vector<std::string> > map_for;
    std::map<std::string,int> map_id_to_index;
    unsigned int box_ids;

private:

    void file_save();
    void file_save_as();
    void remove_selected_images();

    virtual void on_window_resized();
    virtual void on_keydown (
        unsigned long key,
        bool is_printable,
        unsigned long state
    );
    void propagate_labels2(dlib::image_dataset_metadata::dataset& data,unsigned long prev,unsigned long next);
    void on_lb_images_clicked(unsigned long idx); 
    void select_image(unsigned long idx);
    void save_metadata_to_file (const std::string& file);
    void load_image(unsigned long idx);
    void load_image_and_set_size(unsigned long idx);
    void on_overlay_rects_changed();
    void on_overlay_label_changed();
    void on_overlay_rect_selected(const dlib::image_display::overlay_rect& orect);

    void display_about();

    std::string filename;
    dlib::image_dataset_metadata::dataset metadata;

    dlib::menu_bar mbar;
    dlib::list_box lb_images;
    

    dlib::image_display display;
    dlib::label overlay_label_name;
    dlib::text_field overlay_label;
    int lastx;
    int lasty;

    unsigned long keyboard_jump_pos;
    time_t last_keyboard_jump_pos_update;
    
protected:
//    void on_mouse_up (
//                      unsigned long btn,
//                      unsigned long state,
//                      long x,
//                      long y
//                      );
    //void on_mouse_move (unsigned long state,long x,long y);
};

// ----------------------------------------------------------------------------------------


#endif // DLIB_METADATA_EdITOR_H__

