import Header from "../components/Header/Header";

function userLayout({children}){
    return(
        <div>
            <Header />
            {children}
            
        </div>
    )
}
export default userLayout;