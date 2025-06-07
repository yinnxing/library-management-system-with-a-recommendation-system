import Header from "../components/Header/Header";
import Footer from "../components/Footer/Footer";

function userLayout({children}){
    return(
        <div>
            <Header />
            {children}
            <Footer />
        </div>
    )
}
export default userLayout;