/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACTMULTI_PATTERNTREE_H
#define TRAACTMULTI_PATTERNTREE_H

#include "../TraactGuiApp.h"

namespace traact::gui {
    class PatternTree {
    public:
        PatternTree(TraactGuiApp *traactApp);

        void Draw();

    private:
        TraactGuiApp* traact_app_;
    };
}




#endif //TRAACTMULTI_PATTERNTREE_H
